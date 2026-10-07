"""Re-estimate the complete grid with conditional remaining lotteries.

Default order: Test 3, Test 1, Test 2. Original results_grid is read-only.
Pure-reference memory labels are equivalent: fit once, then save an explicitly
labelled alias with the corresponding canonical/free parameter layout.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import copy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import pickle
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import functions as f
from GlobalSettings import GlobalLottery, GlobalSeedsSet, GlobalTol, GlobalInterMax
from main import get_observed_ce
from model_spec import CANON, bounds_free
from grid_em import grid_em_multistart, spec_cluster_layouts, _expand_all, compute_bic
from run_grid import generate_jobs

ROOT = Path(__file__).resolve().parent
VERSION = "conditional_remaining_v1"
SOURCE_FILES = ["functions.py", "Mixture.py", "grid_em.py", "run_conditional_grid.py",
                "run_grid.py", "model_spec.py", "GlobalSettings.py", "lotteries.py",
                "data.py", "main.py", "gamedata.csv"]


def initial_values(result, spec):
    """Project a saved canonical parameter set to a compatible target layout."""
    layouts, identities = spec_cluster_layouts(spec)
    if len(result["thetas"]) != spec.C or result["spec"].pw != spec.pw:
        raise ValueError("Incompatible warm start")
    vectors = []
    ix = CANON[spec.pw]
    for theta, old_delta, layout, identity in zip(result["thetas"], result["delta_les"], layouts, identities):
        theta = np.asarray(theta).copy()
        effective_le = theta[ix["DELTA"]] if old_delta is None else old_delta
        if spec.memory == "commem" and identity == "LE":
            theta[ix["DELTA"]] = effective_le
        vector = [theta[position] for position in layout.free_positions]
        if layout.has_delta_le:
            vector.append(effective_le)
        vectors.append(np.array([np.clip(value, low, high)
                                for value, (low, high) in zip(vector, bounds_free(layout))]))
    return dict(free_vecs=vectors, pis=result["pis"].copy(), ksi=result["ksi"].copy())


def equivalent_result(result, spec):
    clone = copy.deepcopy(result)
    values = initial_values(result, spec)
    clone["spec"] = spec
    clone["layouts"], clone["rp_identities"] = spec_cluster_layouts(spec)
    clone["free_vecs"] = values["free_vecs"]
    clone["thetas"], clone["delta_les"] = _expand_all(clone["free_vecs"], clone["layouts"])
    clone["equivalent_to"] = result["spec"].config_id()
    return clone


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def save(output, spec, result, n_obs, elapsed, fingerprint):
    cid = spec.config_id()
    directory = output / cid
    directory.mkdir(exist_ok=True)
    bic, k = compute_bic(result, n_obs)
    result.update(bic=bic, k_params=k, n_obs=n_obs, run_fingerprint=fingerprint)
    temporary = directory / "result.tmp.pkl"
    with temporary.open("wb") as fh:
        pickle.dump(result, fh)
    temporary.replace(directory / "result.pkl")
    meta = dict(config_id=cid, status="done", **asdict(spec),
        valuation_version=VERSION, run_fingerprint=fingerprint,
        log_likelihood=result["log_likelihood"], k_params=k, bic=bic, n_obs=n_obs,
        n_subjects=len(result["subjects"]), converged=result["converged"],
        exit_reason=result["exit_reason"], n_iter=result["n_iter"],
        n_restarts_ok=result.get("n_restarts_ok"), n_converged=result.get("n_converged"),
        n_random_restarts=result.get("n_random_restarts"), n_warm_starts=result.get("n_warm_starts"),
        seeds_used=list(map(int,result["seeds_used"])), elapsed_seconds=elapsed,
        pis=list(map(float,result["pis"])), rp_identities=result["rp_identities"],
        equivalent_to=result.get("equivalent_to"),
        timestamp=datetime.now(ZoneInfo("Europe/Zurich")).isoformat(),
        optimizer_unsuccessful_calls=sum(not entry["success"] for entry in result["optimizer_log"]),
        optimizer_rejected_steps=sum(not entry["accepted"] for entry in result["optimizer_log"]),
        final_ll_change=result["final_ll_change"])
    atomic_json(directory / "meta.json", meta)
    return meta


def load_completed(output, cid, fingerprint):
    path = output / cid / "meta.json"
    if not path.exists():
        return None
    meta = json.loads(path.read_text())
    if meta.get("run_fingerprint") != fingerprint:
        raise ValueError(f"Existing corrected run has a different fingerprint: {path}")
    if meta["status"] != "done":
        return None
    with (path.parent / "result.pkl").open("rb") as fh:
        return pickle.load(fh)


def write_summary(output):
    records = [json.loads(path.read_text()) for path in output.glob("test*/meta.json")]
    if records:
        pd.DataFrame(records).sort_values(["test", "bic"]).to_csv(output / "summary.csv", index=False)


def ordered_jobs(which):
    jobs = generate_jobs()
    priority = [("qh", "sepalpha", "prelec"), ("exp", "sepalpha", "prelec"),
                ("exp", "sepalpha", "tk"), ("qh", "sepalpha", "tk"),
                ("exp", "commonalpha", "prelec"), ("qh", "commonalpha", "prelec"),
                ("exp", "commonalpha", "tk"), ("qh", "commonalpha", "tk")]
    test_rank = {"test3":0,"test1":1,"test2":2}
    def key(job):
        _, spec = job
        return (test_rank[spec.test], priority.index((spec.discounting,spec.curvature,spec.pw)),
                0 if spec.memory == "commem" else 1, spec.C, spec.rp_identity or "")
    return sorted([job for job in jobs if which == "all" or job[1].test == which], key=key)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--which",choices=["all","test3","test1","test2"],default="all")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results_grid_conditional")
    parser.add_argument("--seeds",type=int,default=20)
    parser.add_argument("--n-workers",type=int,default=20)
    parser.add_argument("--max-iter",type=int,default=GlobalInterMax)
    parser.add_argument("--tol",type=float,default=GlobalTol)
    parser.add_argument("--list",action="store_true")
    args=parser.parse_args()
    jobs=ordered_jobs(args.which)
    if args.list:
        print("\n".join(cid for cid,_ in jobs));return
    output=args.output_dir.resolve()
    if output == (ROOT/"results_grid").resolve():
        raise ValueError("Original results_grid is protected; use a separate output directory")
    output.mkdir(exist_ok=True)
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    protocol=dict(valuation_version=VERSION,source_hashes=hashes,seeds=list(map(int,GlobalSeedsSet[:args.seeds])),
        max_iter=args.max_iter,tol=args.tol,reference_basis="current remaining payoff; original historical benchmark formulas",
        time_origin="current decision, next payment delay 1",noise_scale="original full-lottery spread",
        memory_equivalence="fit 104 distinct models; save 144 configurations with labelled pure-reference aliases")
    fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    protocol["fingerprint"]=fingerprint
    manifest=output/"run_protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("fingerprint") != fingerprint:
        raise ValueError("Output directory contains a different protocol; choose a new directory")
    atomic_json(manifest,protocol)
    snapshot=output/"corrected_source_snapshot";snapshot.mkdir(exist_ok=True)
    for name in SOURCE_FILES:(snapshot/name).write_bytes((ROOT/name).read_bytes())
    observed=get_observed_ce(export_excel=False)
    lotteries=f.transform(GlobalLottery)
    valid=observed[observed.lottery_id.isin(lotteries)].dropna(subset=["ce_observed"])
    seeds=GlobalSeedsSet[:args.seeds]
    print(f"CONDITIONAL REFIT: {len(jobs)} configurations, {valid.participant_label.nunique()} subjects, {len(valid)} CEs; priority Test 3",flush=True)
    current_test=None
    for index,(cid,spec) in enumerate(jobs,1):
        if spec.test != current_test:
            if current_test:
                print(f"TEST COMPLETE: {current_test}",flush=True)
            current_test=spec.test
        existing=load_completed(output,cid,fingerprint)
        if existing is not None:
            print(f"[{index}/{len(jobs)}] {cid}: checkpoint already complete",flush=True);continue
        print(f"[{index}/{len(jobs)}] START {cid}",flush=True)
        if spec.memory == "sepmem" and spec.rp_mode != "mixed":
            source_cid=cid.replace("_sepmem_","_commem_")
            source=load_completed(output,source_cid,fingerprint)
            if source is None:raise RuntimeError(f"Missing equivalent source {source_cid}")
            result=equivalent_result(source,spec)
            meta=save(output,spec,result,len(valid),0,fingerprint)
            print(f"  equivalent alias of {source_cid}, BIC={meta['bic']:.2f}",flush=True)
            write_summary(output);continue
        checkpoint=output/cid/"restarts";checkpoint.mkdir(parents=True,exist_ok=True)
        initializations={}
        legacy_ids=[cid]
        if spec.rp_mode != "mixed":legacy_ids.append(cid.replace("_commem_","_sepmem_"))
        for old_id in legacy_ids:
            path=ROOT/"results_grid"/old_id/"result.pkl"
            if path.exists():
                with path.open("rb") as fh:old=pickle.load(fh)
                if old["subjects"] != sorted(valid.participant_label.unique()):raise ValueError("Warm-start subject mismatch")
                initializations["legacy_"+old_id]=initial_values(old,spec)
        # An already fitted restricted model is a feasible warm start for a
        # broader setting, preventing avoidable nested-model LL reversals.
        candidates=[]
        for path in output.glob("test*/meta.json"):
            meta=json.loads(path.read_text())
            if meta.get("equivalent_to") or meta["pw"] != spec.pw or meta["C"] != spec.C:continue
            same=meta["test"]==spec.test and meta.get("rp_identity")==spec.rp_identity
            fixed_to_mixed=spec.test=="test1" and spec.C==4 and meta["test"]=="test3"
            if same or fixed_to_mixed:
                score=(meta["curvature"]!=spec.curvature,meta["discounting"]!=spec.discounting,meta["memory"]!=spec.memory,meta["bic"])
                candidates.append((score,meta["config_id"]))
        for _,source_id in sorted(candidates)[:2]:
            source=load_completed(output,source_id,fingerprint)
            initializations["corrected_"+source_id]=initial_values(source,spec)
        started=time.time()
        result=grid_em_multistart(spec,seeds,y=observed,lotteries=lotteries,max_iter=args.max_iter,
            tol=args.tol,n_workers=args.n_workers,initializations=initializations,checkpoint_dir=checkpoint)
        meta=save(output,spec,result,len(valid),time.time()-started,fingerprint)
        write_summary(output)
        print(f"  SAVED {cid}: LL={meta['log_likelihood']:.6f}, BIC={meta['bic']:.2f}, exit={meta['exit_reason']}",flush=True)
    if current_test:print(f"TEST COMPLETE: {current_test}",flush=True)
    print("ALL REQUESTED CONDITIONAL REFITS COMPLETE",flush=True)


if __name__ == "__main__":
    main()
