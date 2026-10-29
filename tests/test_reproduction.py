"""Bitwise reproduction of saved estimates and of the pre-refactor EM.

tests/fixtures/em_reference.npz was produced by the code before the move to
the estimation package (commit f6f02d0): the same seeds and iteration counts
must give identical bits, not just close values.
"""
import hashlib
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from estimation.config import LOTTERIES, ROOT
from estimation.data import exclude_participants, observed_ce
from estimation.em import grid_em_mixture
from estimation.likelihood import result_loglik
from estimation.lottery_paths import transform
from estimation.prepare import prepare_grid_data
from estimation.spec import ModelSpec
from estimation.store import load_pickle

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "em_reference.npz"
REFERENCE_RUNS = {
    "t3_qh_sep_com_prelec": (ModelSpec("test3", "qh", "sepalpha", "commem", "prelec", "fixed_4", 4), 892),
    "t1_exp_sep_sep_tk_C2": (ModelSpec("test1", "exp", "sepalpha", "sepmem", "tk", "mixed", 2), 7739),
}


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lotteries = transform(LOTTERIES)
        cls.observed = observed_ce()
        cls.data = prepare_grid_data(cls.observed, cls.lotteries)

    def test_short_em_runs_match_the_pre_refactor_bits(self):
        reference = np.load(FIXTURE)
        for label, (spec, seed) in REFERENCE_RUNS.items():
            with self.subTest(label=label):
                result = grid_em_mixture(spec, max_iter=4, seed=seed, prepared=self.data)
                for j, vector in enumerate(result["free_vecs"]):
                    np.testing.assert_array_equal(vector, reference[f"{label}/free_vec_{j}"])
                np.testing.assert_array_equal(result["pis"], reference[f"{label}/pis"])
                np.testing.assert_array_equal(result["ksi"], reference[f"{label}/ksi"])
                self.assertEqual(result["log_likelihood"], reference[f"{label}/log_likelihood"])
                np.testing.assert_array_equal([row["log_likelihood"] for row in result["iter_log"]],
                                              reference[f"{label}/iteration_ll"])
                self.assertEqual(hashlib.sha256(np.ascontiguousarray(result["resp"]).tobytes()).hexdigest(),
                                 str(reference[f"{label}/resp_sha256"]))

    def test_saved_results_reproduce_their_likelihood_exactly(self):
        """Every saved model of the current sample (and of the endpoint-exclusion
        samples) recomputes to the identical log-likelihood."""
        directories = [ROOT / "results_grid_conditional_stricter",
                       *sorted(ROOT.glob("results_grid_conditional_stricter_no_endpoint*"))]
        directories = [directory for directory in directories if directory.is_dir()]
        if not directories:
            self.skipTest("Requires saved results_grid_conditional_stricter* directories")
        checked = 0
        for directory in directories:
            data = self.data
            excluded = directory / "excluded_participants.csv"
            if excluded.exists():
                dropped = pd.read_csv(excluded).participant_label
                data = prepare_grid_data(exclude_participants(self.observed, dropped), self.lotteries)
            for path in sorted(directory.glob("test*/result.pkl")):
                with self.subTest(result=str(path.relative_to(ROOT))):
                    result = load_pickle(path)
                    ll, resp = result_loglik(result, data)
                    self.assertEqual(ll, result["log_likelihood"])
                    np.testing.assert_array_equal(resp, result["resp"])
                    checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
