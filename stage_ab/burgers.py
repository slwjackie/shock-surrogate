"""Stage A: H=1 learned flux, offline enclosures, and budgeted fallback.

The existing certified_burgers package is preserved. This module adds a small
2-input ReLU flux for certification, rather than pretending that its certificate
also covers the old 5-input GELU CNN. All certificate arithmetic is fail-closed.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .interval import Interval as I, upper_fraction
from certified_burgers.godunov import godunov_flux, godunov_step
from certified_burgers.verifiers import conservation_defect, weak_residual_score


def exact_godunov(a: Q, b: Q) -> Q:
    """Exact Burgers numerical flux: max((a_+)^2, (b_-)^2)/2."""
    return max(max(a, Q(0))**2, min(b, Q(0))**2)/2


class PairFlux(nn.Module):
    """Two-state numerical flux; supervised flux targets fix the flux gauge."""
    def __init__(self, width=12, dropout=0.05):
        super().__init__()
        self.width = int(width)
        self.net = nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(width, 1))

    def forward(self, pairs):
        return self.net(pairs).squeeze(-1)

    def freeze(self):
        layers = [(m.weight.detach().cpu().double().numpy().copy(),
                   m.bias.detach().cpu().double().numpy().copy())
                  for m in self.net if isinstance(m, nn.Linear)]
        return FrozenFlux(layers)


def train_flux(*, samples=1024, epochs=100, width=12, envelope=2., seed=0):
    if samples < 1 or epochs < 1 or envelope <= 0:
        raise ValueError("samples, epochs and envelope must be positive")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pairs = rng.uniform(-envelope, envelope, (samples, 2)).astype(np.float32)
    target = godunov_flux(pairs[:, 0], pairs[:, 1]).astype(np.float32)
    model = PairFlux(width=width)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    x, y = torch.from_numpy(pairs), torch.from_numpy(target)
    losses = []
    for _ in range(epochs):
        model.train()
        # Deterministic full-batch baseline; architecture tuning is not the study.
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x)-y).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return model, losses


@dataclass
class FrozenFlux:
    layers: list[tuple[np.ndarray, np.ndarray]]
    arithmetic: str = "numpy-float64-separate-multiply-add-v1"

    def __post_init__(self):
        previous = 2
        clean = []
        for w, b in self.layers:
            w, b = np.asarray(w, dtype=np.float64), np.asarray(b, dtype=np.float64)
            if w.ndim != 2 or w.shape[1] != previous or b.shape != (w.shape[0],):
                raise ValueError("Invalid frozen network shape")
            if not (np.isfinite(w).all() and np.isfinite(b).all()):
                raise ValueError("Nonfinite model weights")
            clean.append((w.copy(), b.copy()))
            previous = len(b)
        if not clean or previous != 1:
            raise ValueError("Flux network must have one scalar output")
        self.layers = clean

    def payload(self):
        return {"arithmetic": self.arithmetic,
                "layers": [{"weight": w.tolist(), "bias": b.tolist()} for w, b in self.layers]}

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    def predict(self, pairs):
        """Use precisely the elementary operation order enclosed offline."""
        x = np.asarray(pairs, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
            raise ValueError("Expected finite [batch,2] pairs")
        for k, (w, b) in enumerate(self.layers):
            y = np.broadcast_to(b, (len(x), len(b))).copy()
            for j in range(w.shape[1]):
                product = np.multiply(x[:, j:j+1], w[None, :, j])
                y = np.add(y, product)
            x = np.maximum(y, 0) if k+1 < len(self.layers) else y
        return x[:, 0]

    def enclosure(self, box):
        x = list(box)
        for k, (w, b) in enumerate(self.layers):
            y = []
            for row, bias in zip(w, b):
                accum = I(bias)
                for coefficient, item in zip(row, x):
                    accum = accum + I(coefficient)*item
                y.append(accum.relu() if k+1 < len(self.layers) else accum)
            x = y
        return x[0]

    def real_predict(self, pairs):
        """Exact rational network values, used only for proof/audit tests."""
        out = []
        for pair in pairs:
            x = [Q(float(v)) for v in pair]
            for k, (w, b) in enumerate(self.layers):
                x = [sum((Q(float(a))*v for a, v in zip(row, x)), Q(float(bias)))
                     for row, bias in zip(w, b)]
                if k+1 < len(self.layers):
                    x = [max(Q(0), v) for v in x]
            out.append(x[0])
        return out

    def lipschitz_bound(self):
        """Real ReLU network Lipschitz bound in the infinity norm, exact rational."""
        result = Q(1)
        for w, _ in self.layers:
            result *= max(sum((abs(Q(float(x))) for x in row), Q(0)) for row in w)
        return result

    def fp_roundoff_bound(self, box):
        """Uniform forward-error bound between the frozen FP64 and real network.

        Gamma_(2d) dot-product estimate plus an absolute subnormal allowance;
        component errors propagate through |W| and 1-Lipschitz ReLU.
        """
        x, errors = list(box), [Q(0), Q(0)]
        u, tiny = Q(1, 2**53), Q(1, 2**1074)
        for layer_index, (w, b) in enumerate(self.layers):
            values, new_errors = [], []
            count = 2*w.shape[1]
            gamma = count*u/(1-count*u)
            for row, bias in zip(w, b):
                accum = I(bias)
                magnitude = abs(Q(float(bias)))
                propagated = Q(0)
                for coefficient, item, error in zip(row, x, errors):
                    accum = accum+I(coefficient)*item
                    magnitude += abs(Q(float(coefficient)))*Q(float(item.abs_upper()))
                    propagated += abs(Q(float(coefficient)))*error
                values.append(accum.relu() if layer_index+1 < len(self.layers) else accum)
                new_errors.append(propagated+gamma*magnitude+count*tiny/(1-count*u))
            x, errors = values, new_errors
        return errors[0]

    def save(self, path):
        Path(path).write_text(json.dumps(self.payload(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data["arithmetic"] != "numpy-float64-separate-multiply-add-v1":
            raise ValueError("Unknown certified arithmetic contract")
        return cls([(np.array(x["weight"]), np.array(x["bias"])) for x in data["layers"]])


@dataclass
class FluxCertificate:
    model_hash: str
    edges: np.ndarray
    bounds: np.ndarray
    method: str = "interval"
    build_seconds: float = 0.
    subdivisions: int = 1

    @classmethod
    def build(cls, model, *, envelope=2., bins=8, subdivisions=1, method="interval"):
        if not (np.isfinite(envelope) and envelope > 0 and bins >= 1 and subdivisions >= 1):
            raise ValueError("Invalid certificate partition")
        if method not in {"interval", "lipschitz", "hybrid"}:
            raise ValueError("Unknown certificate construction")
        start = time.perf_counter()
        edges = np.linspace(-envelope, envelope, bins+1)
        bounds = np.zeros((bins, bins))
        for a in range(bins):
            for b in range(bins):
                # Sub-box union is exhaustive; never sample to declare an enclosure.
                ea = np.linspace(edges[a], edges[a+1], subdivisions+1)
                eb = np.linspace(edges[b], edges[b+1], subdivisions+1)
                upper = Q(0)
                for j in range(subdivisions):
                    for k in range(subdivisions):
                        box = [I(ea[j], ea[j+1]), I(eb[k], eb[k+1])]
                        predicted = model.enclosure(box)
                        glo = exact_godunov(Q(box[0].lo), Q(box[1].hi))
                        ghi = exact_godunov(Q(box[0].hi), Q(box[1].lo))
                        interval_bound = max(abs(Q(predicted.lo)-ghi), abs(Q(predicted.hi)-glo))
                        local_bound = interval_bound
                        if method != "interval":
                            center = [(Q(v.lo)+Q(v.hi))/2 for v in box]
                            # Exact center computation avoids rounded-center coverage gaps.
                            x = center
                            for layer_index, (w, biases) in enumerate(model.layers):
                                x = [sum((Q(float(a))*v for a, v in zip(row, x)), Q(float(bias)))
                                     for row, bias in zip(w, biases)]
                                if layer_index+1 < len(model.layers):
                                    x = [max(Q(0), v) for v in x]
                            center_error = abs(x[0]-exact_godunov(center[0], center[1]))
                            radius = max((Q(v.hi)-Q(v.lo))/2 for v in box)
                            M = max(Q(float(v.abs_upper())) for v in box)
                            lipschitz_bound = center_error+(model.lipschitz_bound()+M)*radius+model.fp_roundoff_bound(box)
                            local_bound = lipschitz_bound if method == "lipschitz" else min(interval_bound, lipschitz_bound)
                        upper = max(upper, local_bound)
                bounds[a, b] = upper_fraction(upper)
        return cls(model.digest, edges, bounds, method=method, build_seconds=time.perf_counter()-start, subdivisions=int(subdivisions))

    def lookup(self, model, pairs):
        if model.digest != self.model_hash:
            raise ValueError("Model/certificate fingerprint mismatch")
        x = np.asarray(pairs, dtype=np.float64)
        if not np.isfinite(x).all() or np.any(x < self.edges[0]) or np.any(x > self.edges[-1]):
            raise ValueError("Input outside certified domain")
        ij = np.searchsorted(self.edges, x, side="right")-1
        ij = np.clip(ij, 0, len(self.edges)-2)
        return self.bounds[ij[:, 0], ij[:, 1]]

    def save(self, path):
        data = {"model_hash": self.model_hash, "edges": self.edges.tolist(),
                "bounds": self.bounds.tolist(), "method": self.method,
                "build_seconds": self.build_seconds, "subdivisions": self.subdivisions}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path, model, *, verify=True):
        d = json.loads(Path(path).read_text())
        obj = cls(d["model_hash"], np.array(d["edges"]), np.array(d["bounds"]),
                  d["method"], d["build_seconds"], d["subdivisions"])
        if obj.model_hash != model.digest or obj.method not in {"interval", "lipschitz", "hybrid"}:
            raise ValueError("Certificate contract mismatch")
        n = len(obj.edges)-1
        if (n < 1 or obj.bounds.shape != (n, n) or not np.isfinite(obj.bounds).all()
                or np.any(obj.bounds < 0) or not np.all(np.diff(obj.edges) > 0)):
            raise ValueError("Invalid certificate table")
        # A JSON table is not trusted merely because it contains a model hash.
        # Rebuild on load when requested; a refined table is still verifiable by
        # regenerating the documented subdivision count (stored separately by runner).
        if verify:
            rebuilt = cls.build(model, envelope=float(obj.edges[-1]), bins=n,
                                subdivisions=obj.subdivisions, method=obj.method)
            if not np.array_equal(obj.edges, rebuilt.edges) or np.any(obj.bounds < rebuilt.bounds):
                raise ValueError("Unverified or tampered table; rebuild certificate")
        return obj


def pairs(state):
    return np.column_stack((state, np.roll(state, -1)))


def neural_step(model, state, lam):
    flux = model.predict(pairs(state))
    return state-lam*(flux-np.roll(flux, 1)), flux


def update_roundoff(state, candidate, flux, lam, h):
    """Exact rational error of the floating update, without any Godunov call."""
    qlam, qh = Q(float(lam)), Q(float(h))
    total = Q(0)
    for i in range(len(state)):
        ideal = Q(float(state[i]))-qlam*(Q(float(flux[i]))-Q(float(flux[i-1])))
        total += abs(Q(float(candidate[i]))-ideal)
    return qh*total


def certified_proposal(model, table, state, lam, h):
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 1 or len(state) < 2 or h <= 0 or lam <= 0:
        raise ValueError("Invalid grid/time coefficient")
    delta = table.lookup(model, pairs(state))
    candidate, flux = neural_step(model, state, lam)
    if not np.isfinite(candidate).all():
        raise ArithmeticError("Nonfinite candidate")
    bound = 2*Q(float(lam))*Q(float(h))*sum((Q(float(x)) for x in delta), Q(0))
    rounding = update_roundoff(state, candidate, flux, lam, h)
    return candidate, bound+rounding, rounding


def fallback_step(state, lam, h):
    """Existing FP64 Godunov step plus its exact-arithmetic defect enclosure."""
    candidate, _ = godunov_step(state, dx=1., dt=float(lam))
    qlam, qh = Q(float(lam)), Q(float(h))
    values = [Q(float(x)) for x in state]
    flux = [exact_godunov(values[i], values[(i+1) % len(values)]) for i in range(len(values))]
    error = sum((abs(Q(float(candidate[i]))-(values[i]-qlam*(flux[i]-flux[i-1])))
                 for i in range(len(values))), Q(0))
    return candidate, qh*error


def exact_local_error(candidate, state, lam, h):
    """Research oracle: exact-rational same-state Godunov error."""
    v, p = [Q(float(x)) for x in state], [Q(float(x)) for x in candidate]
    f = [exact_godunov(v[i], v[(i+1) % len(v)]) for i in range(len(v))]
    return Q(float(h))*sum((abs(p[i]-(v[i]-Q(float(lam))*(f[i]-f[i-1])))
                            for i in range(len(v))), Q(0))


def certified_rollout(model, table, initial, *, steps=20, lam=0.2,
                      step_tolerance=0.1, global_budget=1., audit=True):
    state = np.asarray(initial, dtype=np.float64).copy()
    h = 1./len(state)
    envelope = float(max(abs(table.edges[0]), abs(table.edges[-1])))
    if Q(float(lam))*Q(envelope) > 1 or lam <= 0:
        raise ValueError("Certificate requires the common CFL lambda*M <= 1")
    if steps < 0 or min(step_tolerance, global_budget) < 0:
        raise ValueError("Invalid step count/budget")
    table.lookup(model, pairs(state))
    reference = state.copy()
    budget, reference_roundoff = Q(0), Q(0)
    rows = []
    start = time.perf_counter()
    for n in range(steps):
        before = state.copy()
        reason, accept, bound = "unavailable_certificate", False, None
        try:
            candidate, bound, rounding = certified_proposal(model, table, before, lam, h)
            valid = np.isfinite(candidate).all() and np.max(np.abs(candidate)) <= envelope
            accept = valid and bound <= Q(float(step_tolerance)) and budget+bound <= Q(float(global_budget))
            reason = "certified_accept" if accept else ("state_envelope" if not valid else "error_budget")
        except (ValueError, ArithmeticError, OverflowError):
            candidate = None
        if accept:
            state = candidate
            budget += bound
        else:
            state, arithmetic_defect = fallback_step(before, lam, h)
            budget += arithmetic_defect
        row = {"step": n, "accept": accept, "reason": reason,
               "local_bound": None if bound is None else upper_fraction(bound),
               "budget_vs_real_godunov": upper_fraction(budget)}
        if audit:
            if candidate is not None:
                eta = exact_local_error(candidate, before, lam, h)
                row.update(oracle_eta=float(eta), certificate_holds=(bound is not None and eta <= bound))
            reference, defect = fallback_step(reference, lam, h)
            reference_roundoff += defect
            measured = Q(float(h))*sum((abs(Q(float(x))-Q(float(y))) for x, y in zip(state, reference)), Q(0))
            row.update(error_vs_fp64_godunov=float(measured),
                       bound_vs_fp64_godunov=upper_fraction(budget+reference_roundoff),
                       global_bound_holds=measured <= budget+reference_roundoff)
        rows.append(row)
    return {"state": state.tolist(), "rows": rows,
            "acceptance_rate": float(np.mean([r["accept"] for r in rows])) if rows else 0.,
            "final_bound_vs_real_godunov": upper_fraction(budget),
            "audit": audit, "runtime_sec": time.perf_counter()-start,
            "runtime_includes_oracles": audit,
            "claim": "Floating hybrid vs exact-arithmetic same-grid Godunov, common lambda; includes update/fallback roundoff"}


def fourier_counterexample(n=128, frequency=8, amplitude=.05, constant=.5):
    """A verifier-only counterexample, not a claim about a fixed CNN's range."""
    if n < 16 or not 4 < frequency or 2*frequency+4 >= n:
        raise ValueError("Choose 4 < K and 2*K+4 < N to avoid the tested aliases")
    h, dt = 1./n, .2/n
    x = (np.arange(n)+.5)/n
    initial = np.full(n, constant)
    candidate = initial+amplitude*np.sin(2*np.pi*frequency*x)
    return {"n": n, "K": frequency, "amplitude": amplitude,
            "conservation": float(conservation_defect(initial[None], candidate[None], h)[0]),
            "weak_residual": float(weak_residual_score(initial[None], candidate[None], dx=h, dt=dt)[0]),
            "eta": float(h*np.abs(candidate-initial).sum()),
            "scope": "Arbitrary candidate pairs. A deterministic translation-equivariant CNN maps a constant input to a constant output."}
