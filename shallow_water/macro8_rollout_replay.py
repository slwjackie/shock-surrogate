"""One offline replay experiment on states visited by the frozen neural solver.

Starting histories come from an unused tail of the mining distribution. The
teacher labels each incoming predicted field by restarting its eight-step
finite-volume solve at that field. Two equal-budget continuations use either
uniform rows or worst-first rows. Independent evaluation uses unused test and
shifted test trajectory IDs and the full, unchecked Python neural operator.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .macro8_data import Batch, ReferenceSolver, make_split
from .macro8_model import OUT, SEED, TRAJECTORY_TOL, fit, patches, predict

START = 256
MINING_HISTORIES = 128
NEW_PROFILES = 96
PHYSICAL_BLOCKS = 8
TEST_HISTORIES = 32
REPLAY_SEED = 721060


def tail(split, count, solver):
    full = make_split(split, START+count, solver)
    ix = slice(START,START+count)
    return Batch(full.q[ix],full.target[ix],full.families[ix],full.ids[ix])


def good(q):
    return bool(np.isfinite(q).all() and np.all(q[:,0]>0))


def build_pool(solver, base):
    histories = tail("mine", MINING_HISTORIES, solver)
    inp, targets, families, ids, priorities = [], [], [], [], []
    errors, invalid_predictions, unlabelled = [], [], []
    for q0, family, tid in zip(histories.q,histories.families,histories.ids):
        q = q0.copy()
        for k in range(1, PHYSICAL_BLOCKS+1):
            sid = f"{tid}/block-{k}"
            try:
                teacher = solver.advance(q)
            except RuntimeError as err:
                # Finite positive fields can still violate the teacher's CFL
                # constraint. No accurate training label is available here.
                unlabelled.append({"id":sid,"reason":str(err)})
                break
            with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
                candidate = predict(q,base)
            valid = good(candidate)
            err = np.mean(np.abs(candidate-teacher),axis=0) if np.isfinite(candidate).all() else None
            ratio = float(np.max(err/TRAJECTORY_TOL)) if err is not None else None
            physical_failure = int(not valid)
            priorities.append((physical_failure, ratio if ratio is not None else np.inf))
            errors.append(ratio)
            inp.append(q)
            targets.append(teacher)
            ids.append(sid)
            families.append(family)
            if not valid:
                invalid_predictions.append(sid)
                break
            q = candidate
    pool = Batch(np.array(inp), np.array(targets), families, ids)
    meta = dict(trajectory_ids=list(histories.ids), requested_histories=len(histories.ids),
                max_blocks=PHYSICAL_BLOCKS, candidate_rows=len(ids),
                missed_local_accuracy=sum(x is None or x>1 for x in errors),
                nonpositive_or_nonfinite_predictions=invalid_predictions,
                unlabelled_incoming_cfl_or_teacher_failures=unlabelled,
                omitted_rows_due_to_early_exit=len(histories.ids)*PHYSICAL_BLOCKS-len(ids))
    return pool, priorities, meta


def summary(rows):
    completed = [r for r in rows if "terminal_mae" in r]
    a = np.array([r["terminal_mae"] for r in completed])
    return dict(requested=len(rows),survived=len(completed),failed=len(rows)-len(completed),
                terminal_above_target_in_survivors=int(np.sum(np.any(a>TRAJECTORY_TOL,axis=1))) if len(a) else None,
                mean_terminal_mae_in_survivors=a.mean(axis=0).tolist() if len(a) else None,
                worst_terminal_mae_in_survivors=a.max(axis=0).tolist() if len(a) else None)


def evaluate(solver, models, split):
    batch = tail(split, TEST_HISTORIES, solver)
    by_method = {key:[] for key in models}
    for q0, family, tid in zip(batch.q,batch.families,batch.ids):
        teacher = solver.advance(q0,steps=8*PHYSICAL_BLOCKS)
        for name,model in models.items():
            row = dict(id=tid,family=family)
            q = q0.copy()
            for k in range(1,PHYSICAL_BLOCKS+1):
                with np.errstate(invalid="ignore",divide="ignore",over="ignore"):
                    q = predict(q,model)
                if not good(q):
                    row["failed_at_block"] = k
                    row["failure_reason"] = "nonfinite" if not np.isfinite(q).all() else "nonpositive_depth"
                    break
            else:
                row["terminal_mae"] = np.mean(np.abs(q-teacher),axis=0).tolist()
                row["min_final_depth_m"] = float(q[:,0].min())
            by_method[name].append(row)
    common = set.intersection(*[{r["id"] for r in rr if "terminal_mae" in r} for rr in by_method.values()])
    return dict(split=split, start_index=START,
                endpoint="64 steps of same coarse-grid teacher from original q0",
                shared_survivor_count=len(common),
                methods={key:dict(overall=summary(rows),
                                  matched_common_survivors=summary([r for r in rows if r["id"] in common]),
                                  families={f:summary([r for r in rows if r["family"]==f])
                                            for f in sorted(set(batch.families))},cases=rows)
                         for key,rows in by_method.items()})


def main(out):
    primary = OUT/"weights.npz"
    with np.load(primary) as z:
        base = tuple(z[k][2].copy() for k in ("encoder","bias","kernel"))
    train_info = json.loads((OUT/"training.json").read_text())
    with ReferenceSolver() as solver:
        pretrain = make_split("train",256,solver)
        primary_pool = make_split("mine",256,solver)
        selected_before = {tid for tid in train_info["selected"]["hard"]["trajectory_ids"]}
        extra_old = np.array([i for i,tid in enumerate(primary_pool.ids) if tid in selected_before])
        original = tuple(np.concatenate((a,b)) for a,b in zip(
            patches(pretrain,np.arange(len(pretrain.q)),SEED+1),
            patches(primary_pool,extra_old,SEED+4)))
        pool,priority,meta = build_pool(solver,base)
        rank = np.array(sorted(range(len(priority)), key=priority.__getitem__, reverse=True))
        uni_rng, mix_rng = [np.random.default_rng(x) for x in (REPLAY_SEED, REPLAY_SEED+1)]
        uniform = uni_rng.choice(len(rank),NEW_PROFILES,replace=False)
        difficult = rank[:NEW_PROFILES//2]
        difficult = np.r_[difficult,mix_rng.choice(rank[NEW_PROFILES//2:],NEW_PROFILES-len(difficult),replace=False)]
        models = {"previous_hard":base}
        selected = {}
        for name,chosen in (("on_policy_uniform",uniform),("on_policy_hard",difficult)):
            added = patches(pool,chosen,REPLAY_SEED+2)
            inputs = tuple(np.concatenate((a,b)) for a,b in zip(original,added))
            models[name] = fit(base,inputs,REPLAY_SEED+3,cycles=3)
            selected[name] = dict(ids=[pool.ids[i] for i in chosen],
                                  distinct_histories=len({pool.ids[i].split('/')[0] for i in chosen}),
                                  invalid_prediction_examples=sum(int(priority[i][0]) for i in chosen),
                                  above_mae_targets=sum(priority[i][1]>1 for i in chosen))
        out.mkdir(parents=True, exist_ok=True)
        path = out/"replay.npz"
        # Save both networks before generating the held-out evaluation inputs.
        np.savez(path,**{key:np.stack(col) for key,col in zip(("encoder","bias","kernel"),
                  zip(models["on_policy_uniform"],models["on_policy_hard"]))})
        sha = sha256(path.read_bytes()).hexdigest()
        result = dict(source_weights_sha256=sha256(primary.read_bytes()).hexdigest(), weight_sha256=sha,
                      frozen_modes=["on_policy_uniform","on_policy_hard"], initial_model="frozen hard",
                      replay_seed=REPLAY_SEED,new_profiles=NEW_PROFILES,encoder_steps_per_arm=108,
                      readout_solves_per_arm=4, threshold=TRAJECTORY_TOL.tolist(),
                      mining_priority="Nonpositive or nonfinite outgoing proposals first, then descending maximum local field MAE divided by the two thresholds. Retain their preceding valid incoming state and true label. Stop each history at its first invalid output or teacher failure.",
                      mining=meta, selected=selected, splits={})
        for split in ("test","shift"):
            result["splits"][split] = evaluate(solver,models,split)
    (out/"replay.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("mining",{k:v for k,v in meta.items() if k not in ("trajectory_ids","unlabelled_incoming_cfl_or_teacher_failures","nonpositive_or_nonfinite_predictions")})
    print("unlabelled",len(meta["unlabelled_incoming_cfl_or_teacher_failures"]),"invalid proposals",len(meta["nonpositive_or_nonfinite_predictions"]))
    for name,arm in selected.items():
        print("arm",name,{k:v for k,v in arm.items() if k!="ids"})
    for name,dist in result["splits"].items():
        print(name,{k:v['overall'] for k,v in dist["methods"].items()})


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    main(parser.parse_args().out)
