"""Restricted, fail-closed interval certificate for ideal-gas H2 ODE slabs.

Supports NASA7 + elementary/three-body Arrhenius + Troe/Lindemann falloff,
integer mass action, a fixed density and a tube contained in ONE NASA branch.
Rates/thermo coefficients are the binary64 constants exported by Cantera.
The certificate concerns the exact ODE defined by these constants, not physical
mechanism error and not CVODES' numerical error. Unsupported rate laws, empty
third-body/log domains, branch-crossing tubes and failed tube closure reject.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .interval import Interval as I, Dual, value
from .chemistry import cantera


class IntervalKinetics:
    def __init__(self, chemistry):
        ct = cantera()
        self.chemistry = chemistry
        self.gas = chemistry.gas
        self.R = float(ct.gas_constant)
        self.W = [float(x) for x in chemistry.W]
        self.names = chemistry.names
        self.pref = float(self.gas.reference_pressure)
        self.thermo = []
        for species in self.gas.species():
            if not isinstance(species.thermo, ct.NasaPoly2):
                raise ValueError("Interval kinetics requires NASA7/NasaPoly2")
            if species.thermo.reference_pressure != self.pref:
                raise ValueError("Species reference pressures must match the phase")
            self.thermo.append((species.thermo.coeffs.copy(),
                                species.thermo.min_temp, species.thermo.max_temp))
        self.reactions = self.gas.reactions()
        if any(self.gas.multiplier(k) != 1 for k in range(self.gas.n_reactions)):
            raise ValueError("Modified reaction multipliers are not certified")
        for r in self.reactions:
            if r.orders or any(float(n) != int(n) for n in list(r.reactants.values())+list(r.products.values())):
                raise ValueError("Only integer elementary mass action is certified")
            if type(r.rate).__name__ not in {"ArrheniusRate", "TroeRate", "LindemannRate"}:
                raise ValueError(f"Unsupported certified rate {type(r.rate).__name__}")

    def arrhenius(self, rate, T):
        if rate.pre_exponential_factor <= 0:
            raise ArithmeticError("Nonpositive Arrhenius prefactor")
        return rate.pre_exponential_factor*((T.log()*rate.temperature_exponent
                    -rate.activation_energy/(self.R*T)).exp())

    def rhs(self, z, rho):
        T, Y = z[0], z[1:]
        tv = value(T)
        cp, h, s = [], [], []
        for coeffs, minimum, maximum in self.thermo:
            middle = float(coeffs[0])
            if tv.lo < minimum or tv.hi > maximum:
                raise ArithmeticError("Tube outside NASA validity range")
            if tv.lo <= middle <= tv.hi:
                raise ArithmeticError("Tube crosses a NASA polynomial breakpoint")
            a = [float(x) for x in (coeffs[8:] if tv.hi < middle else coeffs[1:8])]
            cp.append(a[0]+a[1]*T+a[2]*T**2+a[3]*T**3+a[4]*T**4)
            h.append(a[0]+a[1]*T/2+a[2]*T**2/3+a[3]*T**3/4+a[4]*T**4/5+a[5]/T)
            s.append(a[0]*T.log()+a[1]*T+a[2]*T**2/2+a[3]*T**3/3+a[4]*T**4/4+a[6])
        gibbs = [a-b for a, b in zip(h, s)]
        concentrations = [rho*y/w for y, w in zip(Y, self.W)]
        zero = T*0.
        production = [zero for _ in Y]
        index = {name: k for k, name in enumerate(self.names)}
        for reaction in self.reactions:
            rate = reaction.rate
            collider = None
            if reaction.third_body is not None:
                body = reaction.third_body
                collider = sum((body.efficiencies.get(name, body.default_efficiency)*c
                                for name, c in zip(self.names, concentrations)), zero)
            if type(rate).__name__ == "ArrheniusRate":
                kf = self.arrhenius(rate, T)
                if collider is not None:
                    kf = kf*collider
            else:
                kinf = self.arrhenius(rate.high_rate, T)
                kzero = self.arrhenius(rate.low_rate, T)
                reduced = kzero*collider/kinf
                kf = kinf*reduced/(1+reduced)
                if type(rate).__name__ == "TroeRate":
                    coeff = rate.falloff_coeffs
                    a, t3, t1 = [float(x) for x in coeff[:3]]
                    if t1 <= 0 or t3 <= 0:
                        raise ArithmeticError("Unsupported zero Troe temperature")
                    center = (1-a)*(-T/t3).exp()+a*(-T/t1).exp()
                    if len(coeff) == 4:
                        center = center+(-float(coeff[3])/T).exp()
                    ln10 = I(10).log()
                    logfc = center.log()/ln10
                    logpr = reduced.log()/ln10
                    c, n = -.4-.67*logfc, .75-1.27*logfc
                    ratio = (logpr+c)/(n-.14*(logpr+c))
                    kf = kf*(ln10*logfc/(1+ratio**2)).exp()
            reactants = {index[k]: int(v) for k, v in reaction.reactants.items()}
            products = {index[k]: int(v) for k, v in reaction.products.items()}
            forward = kf
            for k, power in reactants.items():
                forward = forward*concentrations[k]**power
            reverse = zero
            if reaction.reversible:
                delta_g = sum((v*gibbs[k] for k, v in products.items()), zero)-sum(
                    (v*gibbs[k] for k, v in reactants.items()), zero)
                delta_n = sum(products.values())-sum(reactants.values())
                kc = (-delta_g).exp()*(self.pref/(self.R*T))**delta_n
                reverse = kf/kc
                for k, power in products.items():
                    reverse = reverse*concentrations[k]**power
            progress = forward-reverse
            for k in set(reactants) | set(products):
                production[k] = production[k]+(products.get(k, 0)-reactants.get(k, 0))*progress
        dy = [w*rate/rho for w, rate in zip(self.W, production)]
        cv = sum((y*self.R*(c-1)/w for y, c, w in zip(Y, cp, self.W)), zero)
        if value(cv).lo <= 0:
            raise ArithmeticError("Tube does not certify positive heat capacity")
        thermal = -sum((self.R*T*(hk-1)*rate for hk, rate in zip(h, production)), zero)/(rho*cv)
        return [thermal]+dy

    def jacobian_enclosure(self, box, rho):
        duals = [Dual.variable(x, k, len(box)) for k, x in enumerate(box)]
        result = self.rhs(duals, I(rho))
        return [[entry for entry in row.grad] for row in result]


def weighted_log_norm_inf(jacobian, scales):
    """Certified upper bound on mu_inf(D^-1 J D), NOT spectral radius."""
    result = -math.inf
    for i, row in enumerate(jacobian):
        bound = I(row[i].hi)
        for j, entry in enumerate(row):
            if i != j:
                bound = bound+I(entry.abs_upper())*I(float(scales[j]))/I(float(scales[i]))
        result = max(result, bound.hi)
    return result


def growth_bound(initial_error, residual_upper, mu_upper, dt):
    """Outward Gronwall envelope for e' <= mu e+r (nonnegative e,r)."""
    if min(initial_error, residual_upper, dt) < 0 or not all(map(math.isfinite,
               [initial_error, residual_upper, mu_upper, dt])):
        raise ValueError("Invalid growth-bound arguments")
    if dt == 0:
        return float(initial_error)
    exponent = (I(mu_upper)*I(dt)).exp()
    factor = I(dt) if mu_upper == 0 else (exponent-I(1))/I(mu_upper)
    result = exponent*I(initial_error)+factor*I(residual_upper)
    return max(0., result.hi)


@dataclass(frozen=True)
class ODECertificate:
    available: bool
    bound: float | None
    reason: str
    slabs: tuple[dict, ...] = ()
    target: str = "exact constant-volume kinetics ODE, not Cantera numerical trajectory"


def certify_linear_slab(kinetics, initial, candidate, dt, *, scales=None,
                        tube_radius=.01, pieces=2, initial_error=0.):
    """A posteriori interval residual + interval-AD stability + tube closure.

    The exact initial state is initial.vector when initial_error=0. Otherwise
    initial_error is an already PROVEN weighted infinity radius. No sampled
    residual or finite-difference eigenvalue is accepted as rigorous evidence.
    """
    if dt <= 0 or tube_radius <= 0 or pieces < 1 or initial_error < 0:
        raise ValueError("Invalid certification slab")
    if candidate.rho != initial.rho:
        return ODECertificate(False, None, "density_changed")
    z0, z1 = initial.vector, candidate.vector
    scales = np.r_[1000., np.ones(len(initial.Y))] if scales is None else np.asarray(scales)
    if scales.shape != z0.shape or np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("Positive fixed component scales are required")
    rows, error = [], float(initial_error)
    try:
        slope = [(I(float(b))-I(float(a)))/I(float(dt)) for a, b in zip(z0, z1)]
        for k in range(pieces):
            # Exact rational fractions represented by interval division.
            theta = I(float(k))/I(float(pieces))
            theta_end = I(float(k+1))/I(float(pieces))
            span = I(theta.lo, theta_end.hi)
            path = [I(float(a))+(I(float(b))-I(float(a)))*span for a, b in zip(z0, z1)]
            box = [p+I(-tube_radius, tube_radius)*I(float(scale)) for p, scale in zip(path, scales)]
            j = kinetics.jacobian_enclosure(box, initial.rho)
            mu = weighted_log_norm_inf(j, scales)
            rhs_path = kinetics.rhs(path, I(initial.rho))
            residual = max(((d-r)/I(float(scale))).abs_upper()
                           for d, r, scale in zip(slope, rhs_path, scales))
            # An upward duration is conservative for the bound only when mu>=0;
            # use the interval duration directly through the positive growth
            # expression, taking mu>=0 as a conservative choice for closure.
            duration = (I(float(dt))/I(float(pieces))).hi
            mu_for_bound = max(0., mu)
            updated = growth_bound(error, residual, mu_for_bound, duration)
            rows.append({"piece": k, "mu_upper": mu, "residual_upper": residual,
                         "bound": updated, "tube_radius": tube_radius})
            if max(error, updated) >= tube_radius:
                return ODECertificate(False, None, "tube_closure_failed", tuple(rows))
            error = updated
        return ODECertificate(True, error, "interval_residual_and_closed_tube", tuple(rows))
    except (ArithmeticError, ValueError, OverflowError, ZeroDivisionError) as exc:
        return ODECertificate(False, None, str(exc), tuple(rows))


def audit_complete_trajectory(kinetics, states, times, *, scales=None, tube_radius=.01, pieces=1):
    """Validate EVERY slab, including any numerical fallback, or withdraw claim.

    This is an expensive research audit, not the default online chemistry gate.
    A failed slab returns no final bound; no unknown fallback error is set to zero.
    """
    times = np.asarray(times, dtype=float)
    if len(states) != len(times) or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid trajectory audit")
    error, rows = 0., []
    from fractions import Fraction
    for n, dt in enumerate(np.diff(times)):
        if Fraction(float(dt)) != Fraction(float(times[n+1]))-Fraction(float(times[n])):
            return {"available": False, "final_bound": None, "rows": rows,
                    "reason": "Time-grid difference needs a separate rounding enclosure"}
        result = certify_linear_slab(kinetics, states[n], states[n+1], float(dt),
                                    scales=scales, tube_radius=tube_radius,
                                    pieces=pieces, initial_error=error)
        rows.append({"step": n, "available": result.available, "bound": result.bound,
                     "reason": result.reason})
        if not result.available:
            return {"available": False, "final_bound": None, "rows": rows,
                    "reason": "At least one slab (including fallback slabs) is unvalidated"}
        error = result.bound
    return {"available": True, "final_bound": error, "rows": rows,
            "target": "Exact ODE from the identical initial state and fixed-density mechanism"}
