# Legacy pipeline (frozen)

Earlier estimation code, kept runnable but no longer maintained. The current
pipeline is the `estimation` package with the scripts in `scripts/`.

This folder is a self-contained snapshot as of commit `f6f02d0`: the scripts
and the modules they import (`functions.py`, `GlobalSettings.py`,
`lotteries.py`, `data.py`, `main.py`, `model_spec.py`, `grid_em.py`,
`Mixture.py`). Run the scripts from the repository root so that
`gamedata.csv` and the output folders resolve as before:

```bash
python legacy/EM_parallel.py
python legacy/two_step_em.py --method prelec
```

| Script | What it estimated | Superseded by |
|---|---|---|
| `Mixture.py`, `EM_parallel.py` | free reference-point mixture (row-wise likelihood) | `scripts/run_grid.py`, Test 1 |
| `fixed_refpoint_em.py` | one cluster per reference point | `scripts/run_grid.py`, Test 3 |
| `two_step_em.py` | session-1 preferences, then reference-point weights | — |
| `MLE.py`, `MLE_parallel.py` | single-type MLE, forward-looking weight | — |
| `compare_stakes.py` | full vs low-stake lottery sets | — |
| `run_grid.py`, `consolidate_grid.py` | first 144-model grid (`results_grid/`) | `scripts/run_grid.py`, `scripts/consolidate.py` |
| `generate_pilot.py`, `ParameterRecovery/`, `Simulation of CE/` | pseudo data, parameter recovery, CE simulation | — |
