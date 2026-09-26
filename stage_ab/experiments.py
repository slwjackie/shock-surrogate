"""Reproducible CLI: python -m stage_ab.experiments {a,b} --smoke --out PATH."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .burgers import (train_flux, FluxCertificate, FrozenFlux, neural_step,
                      certified_rollout, fourier_counterexample)
from certified_burgers.godunov import godunov_step
from certified_burgers.initial_conditions import sample_states
from certified_burgers.verifiers import conservation_defect, weak_residual_score


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    return value


def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "platform": platform.platform(),
            "torch_threads": torch.get_num_threads()}


def stage_a(out, *, smoke=False, seed=0, epochs=None, bins=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    config = {"seed": seed, "epochs": (8 if smoke else 200) if epochs is None else epochs,
              "bins": bins or (4 if smoke else 16), "width": 8 if smoke else 12,
              "n_cells": 32 if smoke else 128, "steps": 4 if smoke else 24,
              "envelope": 2., "lambda": .2}
    n, steps, lam, h = config["n_cells"], config["steps"], config["lambda"], 1/config["n_cells"]
    start = time.perf_counter()
    model, losses = train_flux(samples=256 if smoke else 4096, epochs=config["epochs"],
                               width=config["width"], envelope=2., seed=seed)
    frozen = model.freeze()
    training_sec = time.perf_counter()-start
    table = FluxCertificate.build(frozen, envelope=2., bins=config["bins"])
    frozen.save(out/"flux_model.json"); table.save(out/"flux_certificate.json")
    calibration = sample_states(8 if smoke else 64, n, seed=seed+1, max_abs=1.)
    sets = {"id": sample_states(2 if smoke else 8, n, seed=seed+2, max_abs=1.),
            "ood": sample_states(2 if smoke else 8, n, seed=seed+3, max_abs=1.8, strong_ood=True)}
    def score(name, state, candidate):
        if name == "conservation": return float(conservation_defect(state[None], candidate[None], h)[0])
        if name == "residual": return float(weak_residual_score(state[None], candidate[None], dx=h, dt=lam*h)[0])
        if name == "uncertainty":
            x = torch.tensor(np.column_stack((state, np.roll(state, -1))), dtype=torch.float32)
            draws = []
            model.train()
            with torch.no_grad():
                for _ in range(4 if smoke else 8):
                    f = model(x).numpy()
                    draws.append(state-lam*(f-np.roll(f, 1)))
            model.eval()
            return float(np.std(draws, axis=0, ddof=1).mean())
        raise ValueError(name)
    calibration_rows = []
    torch.manual_seed(seed)
    for state in calibration:
        candidate, _ = neural_step(frozen, state, lam)
        reference, _ = godunov_step(state, 1., dt=lam)
        calibration_rows.append({"eta": h*np.abs(candidate-reference).sum(),
                                 **{k: score(k, state, candidate) for k in ("conservation", "residual", "uncertainty")}})
    thresholds = {name: float(np.quantile([r[name] for r in calibration_rows], .8))
                  for name in ("eta", "conservation", "residual", "uncertainty")}
    ratio = [r["eta"]/max(r["residual"], 1e-12) for r in calibration_rows]
    empirical_multiplier = float(np.quantile(ratio, .95))
    policies = ("always_solver", "always_neural", "oracle", "conservation", "residual",
                "uncertainty", "empirical_envelope", "certified")
    def run_policy(state, policy, diagnostic, tolerance=None):
        if policy == "certified":
            return certified_rollout(frozen, table, state, steps=steps, lam=lam,
                                     step_tolerance=.05 if tolerance is None else tolerance,
                                     global_budget=steps*(.05 if tolerance is None else tolerance), audit=diagnostic)
        current, rows = state.copy(), []
        start = time.perf_counter()
        for step in range(steps):
            reference = None
            if policy == "always_solver":
                current, _ = godunov_step(current, 1., dt=lam)
                rows.append({"accept": False}); continue
            candidate, _ = neural_step(frozen, current, lam)
            if diagnostic or policy == "oracle": reference, _ = godunov_step(current, 1., dt=lam)
            valid = np.isfinite(candidate).all() and np.max(np.abs(candidate)) <= 2.
            if policy == "always_neural": accept = valid
            elif policy == "oracle": accept = valid and h*np.abs(candidate-reference).sum() <= thresholds["eta"]
            elif policy == "empirical_envelope":
                accept = valid and empirical_multiplier*max(score("residual", current, candidate), 1e-12) <= .05
            else: accept = valid and score(policy, current, candidate) <= thresholds[policy]
            row = {"accept": bool(accept)}
            if diagnostic:
                eta = h*np.abs(candidate-reference).sum()
                row.update(eta=float(eta), unsafe_accept=bool(accept and eta > .05))
            current = candidate if accept else (reference if reference is not None else godunov_step(current, 1., dt=lam)[0])
            rows.append(row)
        return {"state": current.tolist(), "rows": rows, "runtime_sec": time.perf_counter()-start,
                "acceptance_rate": float(np.mean([r["accept"] for r in rows]))}
    results, sweeps, all_holds = {}, {}, True
    for role, states in sets.items():
        results[role] = {}
        references = [run_policy(s, "always_solver", False) for s in states]
        for policy in policies:
            rows = []
            for j, state in enumerate(states):
                torch.manual_seed(seed+j)
                diagnostic = run_policy(state, policy, True)
                # Replay the same stochastic decision seed in the timed path.
                times = []
                for _ in range(2):
                    torch.manual_seed(seed+j)
                    timed = run_policy(state, policy, False)
                    times.append(timed["runtime_sec"])
                error = h*np.abs(np.array(diagnostic["state"])-references[j]["state"]).sum()
                row = {"case": j, "error": float(error), "acceptance": diagnostic["acceptance_rate"],
                       "runtime_sec": float(np.median(times)), "steps": diagnostic["rows"]}
                if policy == "certified":
                    row["global_certificate_holds"] = all(x["global_bound_holds"] for x in diagnostic["rows"])
                    all_holds &= row["global_certificate_holds"]
                rows.append(row)
            results[role][policy] = rows
        baseline_time = sum(r["runtime_sec"] for r in results[role]["always_solver"])
        for policy, rows in results[role].items():
            for row in rows:
                row["cohort_speedup"] = baseline_time/sum(r["runtime_sec"] for r in rows)
        sweeps[role] = []
        for tolerance in (.001, .01, .05, .2, 1., 4.):
            runs = [run_policy(s, "certified", True, tolerance) for s in states]
            sweeps[role].append({"step_tolerance": tolerance,
                                "acceptance": float(np.mean([r["acceptance_rate"] for r in runs])),
                                "bound_max": max(r["final_bound_vs_real_godunov"] for r in runs),
                                "all_certificates_hold": all(x["global_bound_holds"] for r in runs for x in r["rows"])})
    payload = {"stage": "A", "config": config, "environment": environment(),
               "training_loss": losses, "training_sec": training_sec,
               "certificate_build_sec": table.build_seconds, "model_sha256": frozen.digest,
               "calibration": calibration_rows, "thresholds": thresholds,
               "empirical_multiplier": empirical_multiplier, "results": results,
               "certified_threshold_sweep": sweeps, "all_global_certificates_hold": all_holds,
               "fourier_counterexample": fourier_counterexample(),
               "claims": ["Certified tiny 2-input ReLU model only, H=1, periodic scalar Burgers.",
                          "No certificate for the old GELU CNN is claimed.",
                          "Certificates include floating-update and fallback rounding, under the documented arithmetic assumptions.",
                          "No exact-PDE-error or performance superiority claim.",
                          "Empirical score/envelope thresholds are calibrated, not certified."]}
    write_json(out/"results.json", payload)
    return payload


def stage_b(out, *, smoke=False, seed=0, epochs=None, mechanism="h2o2.yaml", certified_ode=False, config_path=None):
    from .chemistry import Chemistry, ignition_metrics
    from .hydrogen import (make_cases, generate_data, train_chemistry, local_audit, calibrate,
                           audit_risk, case_initial, rollout, error_scales, state_error, stiffness_diagnostic)
    from .kinetics_bounds import IntervalKinetics, certify_linear_slab
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    chem = Chemistry(mechanism)
    config = {"seed": seed, "epochs": (8 if smoke else 80) if epochs is None else epochs, "smoke": smoke,
              "train_cases": 3 if smoke else 8, "other_cases": 1,
              "points": 20 if smoke else 64, "steps": 8 if smoke else 32,
              "final_time": 3e-4 if smoke else 1e-3, "local_tolerance": .05,
              "certified_ode": certified_ode}
    if config_path is not None:
        changes = json.loads(Path(config_path).read_text())
        allowed = {"epochs", "train_cases", "other_cases", "points", "steps", "final_time", "local_tolerance"}
        if set(changes)-allowed:
            raise ValueError("Unknown Stage B config keys: "+str(set(changes)-allowed))
        config.update(changes)
    if min(config[k] for k in ("epochs", "train_cases", "other_cases", "steps")) < 1:
        raise ValueError("Positive training/case/step counts required")
    write_json(out/"run_config.json", config)
    cases = make_cases(seed, config["train_cases"], config["other_cases"])
    start = time.perf_counter()
    data, manifest, trajectories = generate_data(chem, cases, points=config["points"], final_time=config["final_time"])
    data_sec = time.perf_counter()-start
    # Preserve trajectory IDs and values so data roles are auditable independently.
    arrays = {}
    for key, (t, states) in trajectories.items():
        arrays[key+"_times"] = t
        arrays[key+"_states"] = np.array([s.vector for s in states])
        arrays[key+"_rho"] = np.array([s.rho for s in states])
    np.savez_compressed(out/"reference_trajectories.npz", **arrays)
    write_json(out/"data_manifest.json", {"mechanism": chem.manifest(), "cases": manifest})
    start = time.perf_counter()
    neural, losses = train_chemistry(chem, data["train"], epochs=config["epochs"], width=24 if smoke else 48, seed=seed)
    training_sec = time.perf_counter()-start
    neural.save(out/"chemistry_model.json")
    audits = {role: local_audit(chem, neural, rows, max_rows=8 if smoke else 48, seed=seed+k)
              for k, (role, rows) in enumerate(data.items()) if role != "train"}
    thresholds = calibrate(audits["calibration"], local_tolerance=config["local_tolerance"])
    risk = {role: {name: audit_risk(rows, name, thresholds[name], config["local_tolerance"])
                   for name in thresholds} for role, rows in audits.items()}
    times = np.linspace(0, config["final_time"], config["steps"]+1)
    policies = ["always_solver", "always_solver_restarted", "always_neural", "physical",
                "oracle", "residual", "consistency", "uncertainty"]
    if certified_ode: policies.append("certified_ode")
    results = {}
    for case in cases:
        if case.role in {"train", "calibration"}: continue
        initial, shock = case_initial(chem, case)
        reference = rollout(chem, neural, initial, times, policy="always_solver")
        reference_states = list(reference["states"])
        results[case.case_id] = {"role": case.role, "shock": shock, "policies": {}}
        for policy in policies:
            if policy == "always_solver": diagnostic = dict(reference)
            else:
                diagnostic = rollout(chem, neural, initial, times, policy=policy, threshold=thresholds.get(policy, 0.),
                                     local_tolerance=config["local_tolerance"], diagnostics=True, seed=seed)
            elapsed = []
            # Include NN, projection/EOS, verifier, fallback and reset costs.
            for _ in range(2):
                timed = rollout(chem, neural, initial, times, policy=policy, threshold=thresholds.get(policy, 0.),
                                local_tolerance=config["local_tolerance"], diagnostics=False, seed=seed)
                elapsed.append(timed["runtime_sec"])
            states = diagnostic.pop("states")
            errors = [state_error(a, b, error_scales(chem)) for a, b in zip(states, reference_states)]
            row = {**diagnostic, "runtime_sec": float(np.median(elapsed)),
                   "max_scaled_state_error": float(max(errors)), "trajectory": [s.vector.tolist() for s in states],
                   "density": [s.rho for s in states], "pressure_Pa": [chem.pressure(s) for s in states],
                   "heat_release": [chem.heat_release(s) for s in states]}
            results[case.case_id]["policies"][policy] = row
        policy_rows = results[case.case_id]["policies"]
        reference_delay = policy_rows["always_solver"]["ignition"]["temperature_threshold_delay_s"]
        baseline_runtime = min(policy_rows[k]["runtime_sec"] for k in ("always_solver", "always_solver_restarted"))
        for policy, row in policy_rows.items():
            delay = row["ignition"]["temperature_threshold_delay_s"]
            row["ignition_delay_absolute_error_s"] = abs(delay-reference_delay) if delay is not None and reference_delay is not None else None
            row["ignition_classification_correct"] = (delay is None) == (reference_delay is None)
            row["speedup_vs_fastest_classical"] = baseline_runtime/row["runtime_sec"]
        results[case.case_id]["times_s"] = times.tolist()
        write_json(out/"cases"/(case.case_id+".json"), results[case.case_id])
        print("completed "+case.case_id, flush=True)
    # Independent-tolerance check, not mechanism/experimental validation.
    check_initial, _ = case_initial(chem, cases[0])
    base = chem.trajectory(check_initial, times)
    tighter = chem.trajectory(check_initial, times, rtol=1e-12, atol=1e-22)
    tolerance_check = max(state_error(a, b, error_scales(chem)) for a, b in zip(base, tighter))
    dense_times = np.linspace(0, config["final_time"], 2*config["steps"]+1)
    dense = chem.trajectory(check_initial, dense_times, rtol=1e-12, atol=1e-22)
    coarse_ignition = ignition_metrics(times, tighter, chem.names)
    dense_ignition = ignition_metrics(dense_times, dense, chem.names)
    grid_check = {"coarse": coarse_ignition, "twice_dense": dense_ignition}
    tiny = chem.fresh(T=1100., p=1e5)
    tiny_end = chem.step(tiny, 1e-10)
    try:
        formal_probe = asdict(certify_linear_slab(IntervalKinetics(chem), tiny, tiny_end, 1e-10, tube_radius=1e-5, pieces=1))
    except ValueError as exc:
        formal_probe = {"available": False, "bound": None, "reason": str(exc)}
    payload = {"stage": "B", "config": config, "environment": environment(), "mechanism": chem.manifest(),
               "data_generation_sec": data_sec, "training_sec": training_sec, "training_loss": losses,
               "thresholds": thresholds, "local_audits": audits, "held_out_risk": risk, "results": results,
               "reference_tolerance_check_max_scaled_difference": tolerance_check,
               "ignition_time_sampling_check": grid_check,
               "stiffness_diagnostic": stiffness_diagnostic(chem, check_initial, float(times[1])),
               "restricted_exact_ode_certificate_probe": formal_probe,
               "claims": ["0-D H2 kinetics/autoignition and frozen-shock-conditioned ignition only; no reacting-flow CFD.",
                          "Default gates are empirical; physical projection is validated numerically, not formally.",
                          "Restricted interval ODE certificate fails closed outside supported mechanism/tube assumptions.",
                          "Cantera fallback is not a validated ODE solver; local certified proposals do not by themselves certify the whole rollout.",
                          "Always-neural is guard-limited, not allowed to install an invalid state.",
                          "Benchmarks compare against the faster persistent/restarted Cantera baseline, including all hybrid overhead.",
                          "Short training/smoke results are correctness tests, not publication-level performance evidence."]}
    write_json(out/"results.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["a", "b"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--bins", type=int)
    parser.add_argument("--mechanism", default="h2o2.yaml")
    parser.add_argument("--certified-ode", action="store_true")
    parser.add_argument("--config", help="Stage B JSON overrides (see configs/stage_ab)")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "a":
        payload = stage_a(args.out, smoke=args.smoke, seed=args.seed, epochs=args.epochs, bins=args.bins)
    else:
        payload = stage_b(args.out, smoke=args.smoke, seed=args.seed, epochs=args.epochs,
                          mechanism=args.mechanism, certified_ode=args.certified_ode, config_path=args.config)
    print(json.dumps({"stage": payload["stage"], "out": str(Path(args.out).resolve()), "claims": payload["claims"]}, indent=2))


if __name__ == "__main__":
    main()
