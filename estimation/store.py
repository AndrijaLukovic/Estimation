"""Result files: atomic writes, run protocols and fingerprints, saved models.

A run protocol's fingerprint covers the scientific inputs only (data file,
estimation sample, lottery set, bounds, noise bounds, R^SQ mode, valuation
version, seeds, tolerances, sample filters). Code provenance (git commit, package hash) is
recorded next to it but does not change the fingerprint; reproducibility of
saved results is checked by recomputing their likelihood instead.
"""
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
from zoneinfo import ZoneInfo

import pandas as pd

from .config import BOUNDS, DATA_FILE, KSI_BOUNDS, LOTTERIES, ROOT, RSQ_MODE, VALUATION_VERSION
from .likelihood import compute_bic


def atomic_write_bytes(path, data):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.tmp{os.getpid()}")
    temporary.write_bytes(data)
    temporary.replace(path)


def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_json(path, value, **options):
    atomic_write_text(path, json.dumps(value, **{"ensure_ascii": False, "indent": 2, **options}))


def atomic_pickle(path, value):
    atomic_write_bytes(path, pickle.dumps(value))


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def load_result(path):
    """A saved conditional Grid EM result (result.pkl or restart checkpoint)."""
    result = load_pickle(path)
    if result.get("valuation_version") != VALUATION_VERSION or "spec" not in result:
        raise ValueError("Use a saved conditional Grid EM result.pkl, not a legacy EM result")
    return result


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


# ── Protocols ─────────────────────────────────────────────────────────────

SAMPLE_COLUMNS = ["participant_label", "lottery_id", "round_number", "ce_observed",
                  "realized_period1_label", "realized_period2_label"]
# Inputs that change the likelihood at given parameters (bounds only change the search).
LIKELIHOOD_INPUTS = ("data_sha256", "sample_sha256", "lotteries_sha256", "rsq_mode", "valuation_version")


def sample_sha256(observed):
    """Hash of the estimation sample: whose CEs, for which lotteries and histories."""
    return sha256_bytes(observed[SAMPLE_COLUMNS].to_csv(index=False).encode())


def scientific_inputs(observed, bounds=None, data_file=DATA_FILE):
    """Inputs that determine estimates, independent of how the code is organised.
    ``observed`` is the estimation sample (after any participant exclusions)."""
    return dict(
        data_sha256=sha256_file(data_file), sample_sha256=sample_sha256(observed),
        lotteries_sha256=sha256_bytes(json.dumps(LOTTERIES, sort_keys=True).encode()),
        bounds={method: [list(b) for b in values] for method, values in (BOUNDS if bounds is None else bounds).items()},
        ksi_bounds=list(KSI_BOUNDS), rsq_mode=RSQ_MODE, valuation_version=VALUATION_VERSION)


def input_differences(protocol, observed):
    """Likelihood inputs of a protocol that differ from the current data and
    settings. Protocols written before the scientific fingerprint only record
    file hashes; for those only the data file can be compared."""
    if "inputs" in protocol:
        current = scientific_inputs(observed)
        return [key for key in LIKELIHOOD_INPUTS if protocol["inputs"].get(key) != current[key]]
    recorded = protocol.get("source_hashes", {}).get(DATA_FILE.name)
    if recorded is None:
        return [DATA_FILE.name]
    return [] if recorded == sha256_file(DATA_FILE) else [DATA_FILE.name]


def provenance():
    """Code version that produced a run (recorded, not fingerprinted)."""
    files = sorted([*ROOT.glob("estimation/**/*.py"), *ROOT.glob("scripts/*.py")])
    package = hashlib.sha256()
    for path in files:
        package.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())

    def git(*arguments):
        try:
            return subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain", "--", "estimation", "scripts")
    return dict(git_commit=git("rev-parse", "HEAD"),
                code_modified=None if status is None else bool(status),
                package_sha256=package.hexdigest())


def fingerprint(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def seal_protocol(protocol):
    """Add the fingerprint of the scientific protocol, then code provenance."""
    return dict(protocol, fingerprint=fingerprint(protocol), provenance=provenance())


def write_protocol(output, protocol, name="run_protocol.json"):
    """Allow a failed preflight to retry, but protect existing estimation files."""
    manifest = output / name
    if manifest.exists() and json.loads(manifest.read_text()).get("fingerprint") != protocol["fingerprint"]:
        if any(path.is_file() for path in output.glob("test*/**/*")):
            raise ValueError("Output directory contains a different protocol; choose a new directory")
        print("Updating protocol: output directory has no estimation files yet.", flush=True)
    atomic_json(manifest, protocol)


def snapshot_sources(output, extra_files=()):
    """Copy the estimation package, the running script(s) and the data file."""
    snapshot = output / "source_snapshot"
    for path in [*ROOT.glob("estimation/**/*.py"), *map(Path, extra_files), DATA_FILE]:
        target = snapshot / Path(path).resolve().relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(path).read_bytes())


# ── Saved models ──────────────────────────────────────────────────────────

def save_model(output, spec, result, n_obs, elapsed, fingerprint):
    cid = spec.config_id()
    directory = output / cid
    directory.mkdir(exist_ok=True)
    bic, k = compute_bic(result, n_obs)
    result.update(bic=bic, k_params=k, n_obs=n_obs, run_fingerprint=fingerprint)
    atomic_pickle(directory / "result.pkl", result)
    meta = dict(config_id=cid, status="done", **asdict(spec),
        valuation_version=VALUATION_VERSION, run_fingerprint=fingerprint,
        log_likelihood=result["log_likelihood"], k_params=k, bic=bic, n_obs=n_obs,
        n_subjects=len(result["subjects"]), converged=result["converged"],
        exit_reason=result["exit_reason"], n_iter=result["n_iter"],
        n_restarts_ok=result.get("n_restarts_ok"), n_converged=result.get("n_converged"),
        n_random_restarts=result.get("n_random_restarts"), n_warm_starts=result.get("n_warm_starts"),
        seeds_used=list(map(int, result["seeds_used"])), elapsed_seconds=elapsed,
        pis=list(map(float, result["pis"])), rp_identities=result["rp_identities"],
        equivalent_to=result.get("equivalent_to"),
        timestamp=datetime.now(ZoneInfo("Europe/Zurich")).isoformat(),
        optimizer_unsuccessful_calls=sum(not entry["success"] for entry in result["optimizer_log"]),
        optimizer_rejected_steps=sum(not entry["accepted"] for entry in result["optimizer_log"]),
        final_ll_change=result["final_ll_change"])
    atomic_json(directory / "meta.json", meta)
    return meta


def load_completed(output, cid, fingerprint):
    """The saved result of ``cid`` in this run, or None if not finished."""
    path = output / cid / "meta.json"
    if not path.exists():
        return None
    meta = json.loads(path.read_text())
    if meta.get("run_fingerprint") != fingerprint:
        raise ValueError(f"Existing run has a different fingerprint: {path}")
    if meta["status"] != "done":
        return None
    return load_pickle(path.parent / "result.pkl")


def write_summary(output):
    records = [json.loads(path.read_text()) for path in output.glob("test*/meta.json")]
    if records:
        pd.DataFrame(records).sort_values(["test", "bic"]).to_csv(output / "summary.csv", index=False)
