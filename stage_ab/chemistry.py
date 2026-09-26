"""Constant-volume adiabatic H2 kinetics, physical guards and frozen shocks.

No Euler evolution or shock/reaction feedback is solved here. Frozen normal and
reflected shock states provide initial data for a separate homogeneous reactor.
The species list and thermodynamics always come from the selected mechanism.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.linalg import qr


def cantera():
    try:
        import cantera as ct
    except ImportError as exc:
        raise ImportError("Stage B requires cantera==3.2.0; install requirements-stage-ab.txt") from exc
    return ct


@dataclass(frozen=True)
class State:
    T: float
    rho: float
    Y: np.ndarray

    def __post_init__(self):
        y = np.asarray(self.Y, dtype=float).copy()
        if y.ndim != 1:
            raise ValueError("Y must be a species vector")
        y.setflags(write=False)
        object.__setattr__(self, "Y", y)

    @property
    def vector(self):
        return np.r_[self.T, self.Y]


class Chemistry:
    """A non-thread-safe Cantera adapter; use one instance per worker."""
    def __init__(self, mechanism="h2o2.yaml", phase=None):
        ct = cantera()
        self.gas = ct.Solution(mechanism) if phase is None else ct.Solution(mechanism, phase)
        if self.gas.thermo_model != "ideal-gas":
            raise ValueError("This adapter is for ideal-gas constant-volume kinetics")
        self.mechanism, self.phase = str(mechanism), self.gas.name
        self.names = list(self.gas.species_names)
        self.W = self.gas.molecular_weights.copy()
        self.elements = np.array([[self.gas.n_atoms(k, j)*self.gas.atomic_weight(j)/self.W[k]
                                   for k in range(self.gas.n_species)]
                                  for j in range(self.gas.n_elements)])
        a = np.vstack([np.ones(self.gas.n_species), self.elements])
        _, r, piv = qr(a.T, pivoting=True, mode="economic")
        rank = np.linalg.matrix_rank(r)
        self.constraints = a[piv[:rank]]
        paths = [Path(mechanism)]+[Path(d)/mechanism for d in ct.get_data_directories()]
        source = next((p for p in paths if p.is_file()), None)
        if source is None:
            raise ValueError("Need a readable mechanism file for reproducible fingerprinting")
        self.fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
        self.source = str(source)

    def manifest(self):
        return {"mechanism": self.mechanism, "phase": self.phase,
                "sha256": self.fingerprint, "species": self.names,
                "cantera_version": cantera().__version__,
                "temperature_validity_K": [self.gas.min_temp, self.gas.max_temp],
                "reactor": "closed, adiabatic, constant-volume, ideal-gas"}

    def fresh(self, T=1100., p=101325., phi=1., dilution=3.76):
        if min(T, p, phi) <= 0 or dilution < 0:
            raise ValueError("Invalid fresh-mixture conditions")
        if not {"H2", "O2", "N2"}.issubset(self.names):
            raise ValueError("Mechanism must include H2/O2/N2")
        self.gas.TPX = T, p, {"H2": 2*phi, "O2": 1., "N2": dilution}
        return State(float(self.gas.T), float(self.gas.density), self.gas.Y)

    def set(self, s: State):
        if len(s.Y) != len(self.names) or not np.isfinite(s.vector).all() or s.T <= 0 or s.rho <= 0:
            raise ValueError("Invalid thermochemical state")
        # Prevent Cantera's TDY setter from silently normalizing candidate species.
        self.gas.set_unnormalized_mass_fractions(s.Y)
        self.gas.TD = s.T, s.rho

    def energy(self, s):
        self.set(s)
        return float(self.gas.int_energy_mass)

    def pressure(self, s):
        self.set(s)
        return float(self.gas.P)

    def rhs(self, z, rho):
        s = State(float(z[0]), float(rho), np.asarray(z[1:]))
        self.set(s)
        rate = self.gas.net_production_rates
        dy = self.W*rate/rho
        dT = -float(np.dot(self.gas.partial_molar_int_energies, rate))/(rho*self.gas.cv_mass)
        return np.r_[dT, dy]

    def heat_release(self, s):
        self.set(s)
        return {"enthalpy_heat_release_W_m3": float(-np.dot(self.gas.partial_molar_enthalpies,
                                                           self.gas.net_production_rates)),
                "internal_energy_source_W_m3": float(-np.dot(self.gas.partial_molar_int_energies,
                                                              self.gas.net_production_rates))}

    def step(self, s, dt, *, rtol=1e-10, atol=1e-18):
        if dt < 0 or not np.isfinite(dt):
            raise ValueError("dt must be finite and nonnegative")
        if dt == 0:
            return State(s.T, s.rho, s.Y)
        return self.trajectory(s, np.array([0., dt]), rtol=rtol, atol=atol)[-1]

    def trajectory(self, s, times, *, rtol=1e-10, atol=1e-18):
        ct = cantera()
        t = np.asarray(times, dtype=float)
        if len(t) < 1 or t[0] != 0 or np.any(np.diff(t) <= 0):
            raise ValueError("Times must begin at zero and increase strictly")
        self.set(s)
        reactor = ct.IdealGasReactor(self.gas, energy="on", clone=True)
        network = ct.ReactorNet([reactor])
        network.rtol, network.atol = float(rtol), float(atol)
        network.max_steps = 100000
        out = [s]
        for target in t[1:]:
            network.advance(float(target))
            out.append(State(float(reactor.T), float(reactor.density), reactor.phase.Y))
        return out

    def project(self, raw_y, initial):
        """Nearest nonnegative composition with the SAME elemental inventory.

        SLSQP is a numerical projection, not a formal proof. Its residuals and
        correction size are checked by the gate; cost is included in inference.
        Temperature is recovered at fixed initial internal energy and density.
        """
        y = np.asarray(raw_y, dtype=float)
        if y.shape != initial.Y.shape or not np.isfinite(y).all():
            raise ValueError("Invalid proposed composition")
        a, b = self.constraints, self.constraints@initial.Y
        result = minimize(lambda x: .5*np.sum((x-y)**2), initial.Y.copy(),
                          jac=lambda x: x-y, bounds=[(0., 1.)]*len(y),
                          constraints={"type": "eq", "fun": lambda x: a@x-b,
                                       "jac": lambda x: a}, method="SLSQP",
                          options={"ftol": 1e-13, "maxiter": 100})
        if not result.success or np.max(np.abs(a@result.x-b)) > 1e-9 or np.min(result.x) < -1e-12:
            raise ValueError("Element-constrained composition projection failed")
        energy = self.energy(initial)
        self.gas.UVY = energy, 1./initial.rho, result.x
        state = State(float(self.gas.T), initial.rho, self.gas.Y)
        return state, float(np.linalg.norm(state.Y-y))

    def guard(self, initial, candidate, *, mass_tol=1e-9, element_tol=1e-8,
              energy_tol=1e-7, temperature_range=None):
        z = candidate.vector
        if len(candidate.Y) != len(self.names) or not np.isfinite(z).all() or not np.isfinite(candidate.rho):
            return False, {"reason": "nonfinite_or_shape"}
        if temperature_range is None:
            temperature_range = (self.gas.min_temp, self.gas.max_temp)
        defects = {"mass": float(abs(candidate.Y.sum()-1)),
                   "elements": float(np.max(np.abs(self.elements@(candidate.Y-initial.Y)))),
                   "min_Y": float(candidate.Y.min()),
                   "density_relative": float(abs(candidate.rho-initial.rho)/initial.rho)}
        if (candidate.rho <= 0 or defects["density_relative"] > 1e-12 or
                not temperature_range[0] <= candidate.T <= temperature_range[1] or
                defects["min_Y"] < -1e-12 or defects["mass"] > mass_tol or
                defects["elements"] > element_tol):
            return False, {"reason": "physical_admissibility", **defects}
        try:
            e0, e1 = self.energy(initial), self.energy(candidate)
            defects["energy_relative"] = abs(e1-e0)/max(abs(e0), 1e6)
            defects["pressure_Pa"] = self.pressure(candidate)
        except (ValueError, RuntimeError):
            return False, {"reason": "thermodynamic_recovery", **defects}
        ok = defects["energy_relative"] <= energy_tol and defects["pressure_Pa"] > 0
        return ok, {"reason": "valid" if ok else "energy_or_pressure", **defects}


def normal_shock(chem, initial, mach):
    """Frozen, thermally-perfect ideal-gas Rankine-Hugoniot solution.

    Upstream gas is at rest in the laboratory. wave_speed is the incident shock
    speed. No constant-gamma approximation is used in the energy equation.
    """
    if not np.isfinite(mach) or mach <= 1:
        raise ValueError("Incident Mach number must exceed one")
    chem.set(initial)
    p1, h1 = chem.gas.P, chem.gas.enthalpy_mass
    speed = float(mach*chem.gas.sound_speed)
    def downstream(ratio):
        p2 = p1+initial.rho*speed**2*(1.-1./ratio)
        T2 = initial.T*(p2/p1)/ratio
        return State(T2, initial.rho*ratio, initial.Y)
    def residual(ratio):
        chem.set(downstream(ratio))
        return chem.gas.enthalpy_mass+.5*(speed/ratio)**2-h1-.5*speed**2
    scan = np.linspace(1.+max(1e-10, min(1e-5, (mach-1)*.01)), 30., 400)
    left = scan[0]
    fl = residual(left)
    bracket = None
    for right in scan[1:]:
        fr = residual(right)
        if fl*fr <= 0:
            bracket = (left, right)
            break
        left, fl = right, fr
    if bracket is None:
        raise ValueError("No compressive frozen shock root found")
    ratio = brentq(residual, *bracket, xtol=1e-12)
    out = downstream(ratio)
    return out, {"incident_mach": float(mach), "compression_ratio": ratio,
                 "wave_speed_m_s": speed, "downstream_shock_frame_velocity_m_s": speed/ratio,
                 "downstream_lab_velocity_m_s": speed*(1.-1./ratio),
                 "energy_residual_J_kg": residual(ratio),
                 "model": "frozen thermally-perfect normal shock; not reacting Euler"}


def reflected_shock(chem, initial, mach):
    """Incident shock followed by a frozen reflected shock at a rigid wall."""
    incident, meta = normal_shock(chem, initial, mach)
    speed = meta["downstream_lab_velocity_m_s"]
    chem.set(incident)
    sound = float(chem.gas.sound_speed)
    def residual(backward_speed):
        out, info = normal_shock(chem, incident, (speed+backward_speed)/sound)
        return backward_speed-info["downstream_shock_frame_velocity_m_s"]
    lower = max(0., sound-speed)+.001*sound
    upper = max(sound, speed)*2
    while residual(upper) < 0:
        upper *= 2
        if upper > 1e5:
            raise ValueError("Reflected shock root not bracketed")
    backward = brentq(residual, lower, upper, xtol=1e-10)
    out, reflection = normal_shock(chem, incident, (speed+backward)/sound)
    meta.update(reflected_compression_ratio=reflection["compression_ratio"],
                reflected_wave_speed_m_s=-backward,
                reflected_downstream_lab_velocity_m_s=residual(backward),
                reflected_energy_residual_J_kg=reflection["energy_residual_J_kg"],
                model="frozen incident/reflected shocks; independent constant-volume post-reflected ignition")
    return out, meta


def ignition_metrics(times, states, names, *, temperature_rise=400.):
    """Do not relabel a non-igniting/censored trajectory as ignition at t_end."""
    t = np.asarray(times, dtype=float)
    z = np.array([s.vector for s in states])
    target = z[0, 0]+temperature_rise
    hits = np.flatnonzero(z[:, 0] >= target)
    delay = None
    if len(hits):
        k = int(hits[0])
        delay = float(t[k]) if k == 0 else float(t[k-1]+(t[k]-t[k-1])*(target-z[k-1, 0])/(z[k, 0]-z[k-1, 0]))
    gradient = np.gradient(z[:, 0], t) if len(t) > 2 else np.zeros(len(t))
    peak = int(np.argmax(gradient))
    peak_delay = float(t[peak]) if delay is not None and 0 < peak < len(t)-1 else None
    species = {}
    for name in ("OH", "HO2", "H2O2", "H2", "O2", "H2O"):
        if name in names:
            values = z[:, 1+names.index(name)]
            species[name] = {"peak": float(values.max()), "final": float(values[-1])}
    return {"ignited": delay is not None, "temperature_threshold_delay_s": delay,
            "max_dTdt_delay_s": peak_delay, "censored": delay is None,
            "temperature_rise_K": temperature_rise, "peak_temperature_K": float(z[:, 0].max()),
            "species": species}
