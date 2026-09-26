"""Stage B data, train-only normalization, neural chemistry and gated rollout."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .chemistry import Chemistry, State, reflected_shock, ignition_metrics
from .kinetics_bounds import IntervalKinetics, certify_linear_slab


@dataclass(frozen=True)
class Case:
    case_id: str
    role: str
    T: float
    p: float
    phi: float
    dilution: float = 3.76
    reflected_mach: float | None = None


def make_cases(seed=0, train_cases=12, other_cases=3):
    """Non-overlapping initial-condition trajectories, not shuffled snapshots."""
    if min(train_cases, other_cases) < 1:
        raise ValueError("Need positive case counts")
    rng = np.random.default_rng(seed)
    result = []
    roles = {"train": train_cases, "calibration": other_cases, "test_id": other_cases,
             "ood_temperature": other_cases, "ood_pressure": other_cases,
             "ood_phi": other_cases, "ood_shock": other_cases}
    for role, count in roles.items():
        for k in range(count):
            T = rng.uniform(1050, 1300)
            p = rng.uniform(1, 3)*1e5
            phi, dilution = rng.uniform(.7, 1.3), rng.uniform(3.2, 4.5)
            mach = None
            if role == "ood_temperature": T = rng.uniform(1400, 1550)
            if role == "ood_pressure": p = rng.uniform(8, 12)*1e5
            if role == "ood_phi": phi = rng.uniform(.4, .55)
            if role == "ood_shock":
                T, p, mach = 300., 1e5, float(rng.uniform(2.6, 3.1))
            result.append(Case(f"{role}-{k:03d}", role, float(T), float(p),
                               float(phi), float(dilution), mach))
    return result


def case_initial(chem, case):
    state = chem.fresh(case.T, case.p, case.phi, case.dilution)
    shock = None
    if case.reflected_mach is not None:
        state, shock = reflected_shock(chem, state, case.reflected_mach)
    return state, shock


def error_scales(chem):
    species = np.ones(len(chem.names))
    for name in ("H", "O", "OH", "HO2", "H2O2"):
        if name in chem.names:
            species[chem.names.index(name)] = 1e-4
    return np.r_[1000., species]


def state_error(a, b, scales):
    return float(np.max(np.abs(a.vector-b.vector)/scales))


def features(state, dt):
    if dt <= 0 or state.T <= 0 or state.rho <= 0:
        raise ValueError("Invalid flow-map inputs")
    if np.min(state.Y) < -1e-12:
        raise ValueError("Negative input composition")
    return np.r_[np.log(state.T), np.log(state.rho), np.log(dt),
                 np.log(np.maximum(state.Y, 0)+1e-12)]


def generate_data(chem, cases, *, points=80, final_time=1e-3):
    if points < 4 or final_time <= 1e-8:
        raise ValueError("Need points>=4 and final_time>1e-8 s")
    times = np.r_[0., np.geomspace(1e-8, final_time, points-1)]
    datasets, manifest, trajectories = {}, {}, {}
    signatures = set()
    for case in cases:
        signature = (case.T, case.p, case.phi, case.dilution, case.reflected_mach)
        if signature in signatures:
            raise ValueError("Initial-condition leakage: duplicate thermochemical case")
        signatures.add(signature)
        initial, shock = case_initial(chem, case)
        states = chem.trajectory(initial, times)
        z = np.array([s.vector for s in states])
        rows = datasets.setdefault(case.role, [])
        for i in range(len(times)-1):
            for stride in (1, 2, 4):
                j = i+stride
                if j >= len(times): continue
                dt = float(times[j]-times[i])
                target = np.log(np.maximum(states[j].Y, 0)+1e-12)-np.log(np.maximum(states[i].Y, 0)+1e-12)
                rows.append((features(states[i], dt), target, states[i], states[j], dt, case.case_id))
        digest = hashlib.sha256(z.tobytes()+times.tobytes()).hexdigest()
        manifest[case.case_id] = {**asdict(case), "trajectory_sha256": digest,
                                 "ignition": ignition_metrics(times, states, chem.names), "shock": shock}
        trajectories[case.case_id] = (times, states)
    return datasets, manifest, trajectories


class ChemistryMLP(nn.Module):
    def __init__(self, n_species, width=48, dropout=.05):
        super().__init__()
        self.width, self.dropout = int(width), float(dropout)
        self.net = nn.Sequential(nn.Linear(n_species+3, width), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(width, width), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(width, n_species))
    def forward(self, x):
        return self.net(x)


@dataclass
class NeuralChemistry:
    model: ChemistryMLP
    mean: np.ndarray
    std: np.ndarray
    manifest: dict

    def logits(self, state, dt, *, stochastic=False):
        x = (features(state, dt)-self.mean)/self.std
        self.model.train(stochastic)
        with torch.no_grad():
            increment = self.model(torch.tensor(x[None], dtype=torch.float32))[0].numpy()
        self.model.eval()
        return np.log(np.maximum(state.Y, 0)+1e-12)+increment

    def predict(self, chem, state, dt):
        if self.manifest["sha256"] != chem.fingerprint or self.manifest["species"] != chem.names:
            raise ValueError("Chemistry checkpoint/mechanism mismatch")
        logy = self.logits(state, dt)
        # This decoding is part of the proposal, not a silent guard repair.
        clipped = np.clip(logy, -60, 0)
        raw_y = np.maximum(np.exp(clipped)-1e-12, 0)
        if raw_y.sum() <= 0:
            raise ValueError("Empty decoded composition")
        raw_y /= raw_y.sum()
        candidate, correction = chem.project(raw_y, state)
        return candidate, {"projection_l2": correction,
                           "logit_clipped": bool(np.any(clipped != logy))}

    def uncertainty(self, state, dt, samples=8):
        if samples < 2:
            raise ValueError("MC dropout requires at least two draws")
        values = [self.logits(state, dt, stochastic=True) for _ in range(samples)]
        return float(np.std(values, axis=0, ddof=1).mean())

    def save(self, path):
        payload = {"width": self.model.width, "dropout": self.model.dropout,
                   "mean": self.mean.tolist(), "std": self.std.tolist(),
                   "mechanism": self.manifest,
                   "weights": {k: v.detach().tolist() for k, v in self.model.state_dict().items()}}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path, chem):
        data = json.loads(Path(path).read_text())
        if data["mechanism"]["sha256"] != chem.fingerprint or data["mechanism"]["species"] != chem.names:
            raise ValueError("Checkpoint species/mechanism mismatch")
        model = ChemistryMLP(len(chem.names), data["width"], data["dropout"])
        model.load_state_dict({k: torch.tensor(v) for k, v in data["weights"].items()})
        model.eval()
        return cls(model, np.array(data["mean"]), np.array(data["std"]), data["mechanism"])


def train_chemistry(chem, rows, *, epochs=80, width=48, seed=0):
    if not rows or epochs < 1:
        raise ValueError("Training rows and positive epoch count required")
    if any(not row[-1].startswith("train-") for row in rows):
        raise ValueError("Normalization/training must only use train trajectories")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    mean, std = x.mean(axis=0), np.maximum(x.std(axis=0), .1)
    inputs, targets = torch.tensor((x-mean)/std, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    model = ChemistryMLP(len(chem.names), width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    losses = []
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(rows)); total = 0.
        for start in range(0, len(rows), 128):
            indices = order[start:start+128]
            optimizer.zero_grad(set_to_none=True)
            loss = (model(inputs[indices])-targets[indices]).square().mean()
            loss.backward(); optimizer.step()
            total += float(loss.detach())*len(indices)
        losses.append(total/len(rows))
    model.eval()
    return NeuralChemistry(model, mean, std, chem.manifest()), losses


def verifier_score(name, chem, neural, state, candidate, dt, scales, *, mc_samples=8):
    if name == "residual":
        # Three-time-sample linear-path defect is an EMPIRICAL score, not an upper bound.
        slope = (candidate.vector-state.vector)/dt
        values = []
        for fraction in (.211324865405187, .5, .788675134594813):
            z = (1-fraction)*state.vector+fraction*candidate.vector
            values.append(np.max(np.abs(slope-chem.rhs(z, state.rho))/scales))
        return float(dt*max(values))
    if name == "consistency":
        middle, _ = neural.predict(chem, state, dt/2)
        second, _ = neural.predict(chem, middle, dt/2)
        return state_error(candidate, second, scales)
    if name == "uncertainty":
        return neural.uncertainty(state, dt, mc_samples)
    if name == "physical":
        return 0.
    raise ValueError(f"Unknown empirical verifier {name}")


def local_audit(chem, neural, rows, *, max_rows=64, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    chosen = rng.choice(len(rows), min(max_rows, len(rows)), replace=False)
    scales, out = error_scales(chem), []
    for index in chosen:
        _, _, state, _, dt, case_id = rows[index]
        reference = chem.step(state, dt)  # same-state restarted numerical oracle
        row = {"case_id": case_id, "dt": dt, "valid": False}
        try:
            candidate, details = neural.predict(chem, state, dt)
            valid, guard = chem.guard(state, candidate)
            row.update(valid=valid, eta=state_error(candidate, reference, scales), guard=guard,
                       prediction_details=details)
            for name in ("residual", "consistency", "uncertainty"):
                row[name] = verifier_score(name, chem, neural, state, candidate, dt, scales, mc_samples=4) if valid else float("inf")
        except (ValueError, RuntimeError, ArithmeticError):
            row.update(eta=float("inf"), residual=float("inf"), consistency=float("inf"), uncertainty=float("inf"))
        out.append(row)
    return out


def calibrate(audit, *, local_tolerance=.05, target_empirical_risk=.05):
    """Calibration-only threshold choice; no conformal/probabilistic guarantee."""
    thresholds = {}
    for name in ("residual", "consistency", "uncertainty"):
        values = [r[name] for r in audit if r["valid"] and np.isfinite(r[name])]
        best = -1.
        for threshold in sorted(set(values)):
            accepted = [r for r in audit if r["valid"] and r[name] <= threshold]
            risk = np.mean([r["eta"] > local_tolerance for r in accepted])
            if risk <= target_empirical_risk:
                best = threshold
        thresholds[name] = float(best)
    return thresholds


def audit_risk(audit, name, threshold, tolerance):
    accept = np.array([r["valid"] and r[name] <= threshold for r in audit])
    unsafe = np.array([r["eta"] > tolerance for r in audit])
    return {"acceptance": float(accept.mean()), "samples": len(audit),
            "false_accept_given_accept": float(unsafe[accept].mean()) if accept.any() else None,
            "unsafe_accept_fraction": float((accept & unsafe).mean()),
            "unsafe_miss_rate": float(accept[unsafe].mean()) if unsafe.any() else None}


def rollout(chem, neural, initial, times, *, policy="residual", threshold=0.,
            local_tolerance=.05, diagnostics=False, mc_samples=4, seed=0,
            tube_radius=.01, certificate_pieces=1):
    allowed = {"always_solver", "always_solver_restarted", "always_neural", "physical",
               "oracle", "residual", "consistency", "uncertainty", "certified_ode"}
    if policy not in allowed:
        raise ValueError("Unknown chemistry policy")
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Increasing times beginning at zero required")
    torch.manual_seed(seed)
    scales = error_scales(chem)
    start = time.perf_counter()
    if policy == "always_solver":
        states = chem.trajectory(initial, times)
        elapsed = time.perf_counter()-start
        return {"states": states, "rows": [], "runtime_sec": elapsed,
                "acceptance": 0., "reference_mode": "persistent optimized CVODES",
                "ignition": ignition_metrics(times, states, chem.names)}
    kinetics, unsupported = None, None
    if policy == "certified_ode":
        try:
            kinetics = IntervalKinetics(chem)
        except ValueError as exc:
            unsupported = str(exc)
    states, rows = [initial], []
    current = initial
    for n, dt in enumerate(np.diff(times)):
        reference = None
        if diagnostics or policy == "oracle":
            reference = chem.step(current, float(dt))
        accept, reason, score, candidate = False, "always_solver", None, None
        certificate = None
        if policy != "always_solver_restarted":
            try:
                candidate, info = neural.predict(chem, current, float(dt))
                valid, guard = chem.guard(current, candidate)
                if not valid:
                    reason = guard["reason"]
                elif policy == "always_neural" or policy == "physical":
                    accept, reason = True, "physical_only"  # still rejects invalid states
                elif policy == "oracle":
                    score = state_error(candidate, reference, scales)
                    accept = score <= local_tolerance
                    reason = "oracle_decision"
                elif policy == "certified_ode":
                    if kinetics is None:
                        raise ValueError("Unsupported interval mechanism: "+str(unsupported))
                    certificate = certify_linear_slab(kinetics, current, candidate, float(dt),
                                                     scales=scales, tube_radius=tube_radius,
                                                     pieces=certificate_pieces)
                    score = certificate.bound
                    accept = certificate.available and score <= local_tolerance
                    reason = certificate.reason
                else:
                    score = verifier_score(policy, chem, neural, current, candidate, float(dt), scales,
                                           mc_samples=mc_samples)
                    accept = np.isfinite(score) and score <= threshold
                    reason = "empirical_accept" if accept else "empirical_reject"
            except (ValueError, RuntimeError, ArithmeticError, OverflowError) as exc:
                reason = "proposal_or_verifier_failure: "+str(exc)
        row = {"step": n, "time": float(times[n]), "dt": float(dt), "accept": bool(accept),
               "reason": reason, "score": score}
        if certificate is not None:
            row["certificate"] = asdict(certificate)
        if diagnostics and candidate is not None:
            eta = state_error(candidate, reference, scales)
            row.update(eta_vs_restarted_cantera=eta,
                       unsafe_accept=bool(accept and eta > local_tolerance))
        current = candidate if accept else (reference if reference is not None else chem.step(current, float(dt)))
        states.append(current); rows.append(row)
    elapsed = time.perf_counter()-start
    return {"states": states, "rows": rows, "runtime_sec": elapsed,
            "runtime_includes_oracles": bool(diagnostics or policy == "oracle"),
            "acceptance": float(np.mean([r["accept"] for r in rows])),
            "ignition": ignition_metrics(times, states, chem.names),
            "claim": "Local exact-ODE certificates do not certify a global trajectory with unvalidated CVODES fallback"}


def stiffness_diagnostic(chem, state, dt, scales=None):
    """Finite-difference spectral indicator, explicitly NOT a certificate."""
    scales = error_scales(chem) if scales is None else np.asarray(scales,dtype=float)
    z=state.vector; jac=np.empty((len(z),len(z)))
    for k in range(len(z)):
        delta=1e-5 if k==0 else 1e-8
        left,right=z.copy(),z.copy();left[k]-=delta;right[k]+=delta
        jac[:,k]=(chem.rhs(right,state.rho)-chem.rhs(left,state.rho))/(2*delta)
    weighted=jac*scales[None,:]/scales[:,None]
    eigenvalues=np.linalg.eigvals(weighted)
    return {"rigorous":False,"method":"finite differences, diagnostic only",
            "spectral_radius":float(np.max(np.abs(eigenvalues))),
            "dt_spectral_radius":float(dt*np.max(np.abs(eigenvalues))),
            "max_real_eigenvalue":float(np.max(eigenvalues.real)),
            "min_real_eigenvalue":float(np.min(eigenvalues.real)),
            "weighted_log_norm_inf_estimate":float(np.max(np.diag(weighted)+np.sum(np.abs(weighted),axis=1)-np.abs(np.diag(weighted))))}
