"""Numerical regression checks for conditional remaining-payoff estimation."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

import functions as f
from GlobalSettings import GlobalLottery, GlobalKsiBounds
from main import get_observed_ce
from Mixture import _compute_ce_th, compute_log_likelihoods
from grid_em import prepare_grid_data, grid_em_mixture, grid_em_multistart, noise_mstep
from run_conditional_grid import equivalent_result
from model_spec import ModelSpec, all_specs_16, build_cluster_layout, init_free, expand_free


class ConditionalValuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lotteries = f.transform(GlobalLottery)
        cls.y = get_observed_ce(export_excel=False)
        cls.data = prepare_grid_data(cls.y, cls.lotteries)

    def test_node_probability_and_absolute_probability(self):
        current = f.conditional_lottery(self.lotteries["lottery_8"], 2, 425, -100)
        terminal = {stream[-1]: p for path in current["outcomes"].values()
                    for p, stream in path.items()}
        self.assertAlmostEqual(terminal[160], .2)
        self.assertAlmostEqual(terminal[-860], .8)
        self.assertEqual(len(terminal), 2)
        self.assertTrue(all(len(stream) == 2 for path in current["outcomes"].values() for stream in path.values()))
        with self.assertRaises(ValueError):
            f.conditional_lottery(self.lotteries["lottery_8"], 1, 123456, 0)

    def test_linear_benchmark_and_time_origin(self):
        for stage, expected in [(0, 16.22), (1, 72.6), (2, 22.6)]:
            current = f.conditional_lottery(self.lotteries["lottery_6"], stage, -19, 50)
            for method in ["tk", "prelec"]:
                for reference in [0, 16.22, 72.6, -19]:
                    result = f.evaluation(r=0, R=reference, alpha=1, alpha_loss=1,
                        lamb=1, gamma=1, beta=1, palpha=1, psi=1,
                        method=method, lotteries={"l": current})["l"]
                    ce = f.u_inv(result["V"], reference, 1, 1, alpha_loss=1)
                    self.assertAlmostEqual(ce, expected, places=10)
        r, psi = .1, .8
        for stage, expected in [(1, psi*np.exp(-r)*50 + psi*np.exp(-2*r)*22.6),
                                (2, psi*np.exp(-r)*22.6)]:
            current = f.conditional_lottery(self.lotteries["lottery_6"], stage, -19, 50)
            result = f.evaluation(r=r, R=0, alpha=1, alpha_loss=1, lamb=1,
                gamma=1, psi=psi, method="tk", lotteries={"l": current})["l"]
            self.assertAlmostEqual(result["V"], expected, places=10)

    def test_complete_sample_and_all_observed_histories(self):
        self.assertEqual(len(self.data.subjects), 1131)
        self.assertEqual(len(self.data.observed), 20357)
        self.assertEqual(len(self.data.state_y), 40)
        self.assertTrue(np.allclose(self.data.probabilities.sum(axis=1), 1))

    def test_vectorized_prediction_matches_shared_scalar_formula(self):
        rng = np.random.default_rng(781)
        for disc, curvature, memory, method in all_specs_16():
            for rp in ["SQ", "PA", "LE", "FE", None]:
                spec = ModelSpec("test3" if rp else "test1", disc, curvature, memory,
                                 method, "fixed_4" if rp else "mixed", 4)
                layout = build_cluster_layout(spec, rp)
                theta, delta = expand_free(init_free(layout, rng), layout)
                slow = _compute_ce_th(theta, method, *self.data.reference_inputs,
                                      self.lotteries, self.data.state_y, delta_le=delta)
                fast = self.data.predict(theta, method, delta)
                np.testing.assert_allclose(fast, slow, rtol=2e-9, atol=2e-8)
                # Near-boundary probability functions and strongly asymmetric curvature.
                extreme = theta.copy()
                extreme[1:4] = [5, .7, 6]
                extreme[4 if method == "tk" else 5] = .2389325 if method == "tk" else 1e-6
                slow = _compute_ce_th(extreme, method, *self.data.reference_inputs,
                                      self.lotteries, self.data.state_y, delta_le=delta)
                np.testing.assert_allclose(self.data.predict(extreme, method, delta), slow,
                                           rtol=2e-9, atol=2e-8)
                # Initial-stage prices equal the original full-lottery evaluation.
                initial = self.data.reference_inputs[3] == 0
                for index in np.flatnonzero(initial):
                    key = self.data.state_y.iloc[index]["_ckey"]
                    lot = self.lotteries[key[0]]
                    self.assertEqual(self.data.state_y.attrs["_ce_state_groups"][index][2]["outcomes"], lot["outcomes"])

    def test_exact_noise_update_matches_scalar_minimization(self):
        data = SimpleNamespace(observed=np.array([2., 4., 1., 1., 100., 200.]),
            spread=np.ones(6), subject_index=np.repeat(np.arange(3), 2),
            subjects=["a", "b", "c"], counts=np.array([2, 2, 2]))
        mu = np.array([[2., 3.], [3., 4.], [1., 1.], [1., 1.], [0., 0.], [0., 0.]])
        resp = np.array([[.3, .7], [.5, .5], [.8, .2]])
        exact = noise_mstep(resp, mu, data)
        for subject in range(3):
            mask = data.subject_index == subject
            weighted_sum = np.sum(((data.observed[mask, None]-mu[mask]) / data.spread[mask, None])**2 * resp[subject])
            objective = lambda ksi: data.counts[subject]*np.log(ksi) + weighted_sum/(2*ksi**2)
            result = minimize_scalar(objective, bounds=GlobalKsiBounds, method="bounded")
            self.assertAlmostEqual(exact[subject], result.x, delta=1e-5)

    def test_iteration_limit_and_final_likelihood(self):
        spec = ModelSpec("test3", "qh", "sepalpha", "commem", "prelec", "fixed_4", 4)
        result = grid_em_mixture(spec, self.y, self.lotteries, max_iter=1, seed=892,
                                 prepared=self.data)
        self.assertFalse(result["converged"])
        self.assertEqual(result["exit_reason"], "max_iter")
        log_l = compute_log_likelihoods(result["thetas"], result["ksi"], "prelec", 4,
            result["subjects"], self.y, self.lotteries, delta_le_list=result["delta_les"])
        joint = log_l + np.log(result["pis"])[None, :]
        log_sums = logsumexp(joint, axis=1)
        self.assertAlmostEqual(result["log_likelihood"], log_sums.sum(), places=6)
        np.testing.assert_allclose(result["resp"], np.exp(joint-log_sums[:, None]), atol=1e-9)

    def test_multistart_checkpoint_and_equivalent_memory_alias(self):
        spec = ModelSpec("test3", "exp", "commonalpha", "commem", "tk", "fixed_4", 4)
        with TemporaryDirectory() as directory:
            result = grid_em_multistart(spec, [892], self.y, self.lotteries,
                max_iter=1, n_workers=1, verbose=False, checkpoint_dir=directory)
            path = Path(directory)/"seed_892.pkl"
            self.assertTrue(path.is_file())
            modified = path.stat().st_mtime_ns
            resumed = grid_em_multistart(spec, [892], self.y, self.lotteries,
                max_iter=1, n_workers=1, verbose=False, checkpoint_dir=directory)
            self.assertEqual(modified, path.stat().st_mtime_ns)
            self.assertEqual(result["log_likelihood"], resumed["log_likelihood"])
        other = ModelSpec("test3", "exp", "commonalpha", "sepmem", "tk", "fixed_4", 4)
        alias = equivalent_result(result, other)
        self.assertEqual(alias["equivalent_to"], spec.config_id())
        for th_a, d_a, th_b, d_b in zip(result["thetas"],result["delta_les"],alias["thetas"],alias["delta_les"]):
            np.testing.assert_allclose(self.data.predict(th_a,"tk",d_a),self.data.predict(th_b,"tk",d_b))


if __name__ == "__main__":
    unittest.main()
