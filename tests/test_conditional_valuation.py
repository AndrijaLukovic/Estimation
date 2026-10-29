"""Numerical regression checks for conditional remaining-payoff estimation."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import norm

from estimation import cpt
from estimation.config import KSI_BOUNDS, LOTTERIES
from estimation.data import observed_ce
from estimation.em import grid_em_mixture
from estimation.likelihood import noise_mstep
from estimation.lottery_paths import conditional_lottery, transform
from estimation.multistart import grid_em_multistart
from estimation.prepare import prepare_grid_data
from estimation.spec import ModelSpec, all_specs_16, build_cluster_layout, init_free, expand_free
from estimation.valuation import named_parameters, reference_points, scalar_ce
from estimation.warm_starts import equivalent_result


class ConditionalValuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lotteries = transform(LOTTERIES)
        cls.y = observed_ce()
        cls.data = prepare_grid_data(cls.y, cls.lotteries)

    def scalar_states(self, theta, method, delta_le):
        """State CEs from the scalar CPT reference implementation."""
        values = named_parameters(theta, method)
        references = reference_points(theta, method, delta_le, self.data.reference_inputs)
        return np.array([scalar_ce(values, float(reference), lottery, method)[0]
                         for reference, (_, lottery) in zip(references, self.data.states)])

    def test_node_probability_and_absolute_probability(self):
        current = conditional_lottery(self.lotteries["lottery_8"], 2, 425, -100)
        terminal = {stream[-1]: p for path in current["outcomes"].values()
                    for p, stream in path.items()}
        self.assertAlmostEqual(terminal[160], .2)
        self.assertAlmostEqual(terminal[-860], .8)
        self.assertEqual(len(terminal), 2)
        self.assertTrue(all(len(stream) == 2 for path in current["outcomes"].values() for stream in path.values()))
        with self.assertRaises(ValueError):
            conditional_lottery(self.lotteries["lottery_8"], 1, 123456, 0)

    def test_linear_benchmark_and_time_origin(self):
        for stage, expected in [(0, 16.22), (1, 72.6), (2, 22.6)]:
            current = conditional_lottery(self.lotteries["lottery_6"], stage, -19, 50)
            for method in ["tk", "prelec"]:
                for reference in [0, 16.22, 72.6, -19]:
                    result = cpt.evaluation(r=0, R=reference, alpha=1, alpha_loss=1,
                        lamb=1, gamma=1, beta=1, palpha=1, psi=1,
                        method=method, lotteries={"l": current})["l"]
                    ce = cpt.u_inv(result["V"], reference, 1, 1, alpha_loss=1)
                    self.assertAlmostEqual(ce, expected, places=10)
        r, psi = .1, .8
        for stage, expected in [(1, psi*np.exp(-r)*50 + psi*np.exp(-2*r)*22.6),
                                (2, psi*np.exp(-r)*22.6)]:
            current = conditional_lottery(self.lotteries["lottery_6"], stage, -19, 50)
            result = cpt.evaluation(r=r, R=0, alpha=1, alpha_loss=1, lamb=1,
                gamma=1, psi=psi, method="tk", lotteries={"l": current})["l"]
            self.assertAlmostEqual(result["V"], expected, places=10)

    def test_complete_sample_and_all_observed_histories(self):
        self.assertEqual(len(self.data.subjects), 1081)
        self.assertEqual(len(self.data.observed), 19457)
        self.assertEqual(len(self.data.states), 40)
        self.assertTrue(np.allclose(self.data.probabilities.sum(axis=1), 1))

    def test_vectorized_prediction_matches_shared_scalar_formula(self):
        rng = np.random.default_rng(781)
        for disc, curvature, memory, method in all_specs_16():
            for rp in ["SQ", "PA", "LE", "FE", None]:
                spec = ModelSpec("test3" if rp else "test1", disc, curvature, memory,
                                 method, "fixed_4" if rp else "mixed", 4)
                layout = build_cluster_layout(spec, rp)
                theta, delta = expand_free(init_free(layout, rng), layout)
                slow = self.scalar_states(theta, method, delta)
                fast = self.data.predict(theta, method, delta)
                np.testing.assert_allclose(fast, slow, rtol=2e-9, atol=2e-8)
                # Near-boundary probability functions and strongly asymmetric curvature.
                extreme = theta.copy()
                extreme[1:4] = [5, .7, 6]
                extreme[4 if method == "tk" else 5] = .2389325 if method == "tk" else 1e-6
                slow = self.scalar_states(extreme, method, delta)
                np.testing.assert_allclose(self.data.predict(extreme, method, delta), slow,
                                           rtol=2e-9, atol=2e-8)
                # Initial-stage prices equal the original full-lottery evaluation.
                initial = self.data.reference_inputs[3] == 0
                for index in np.flatnonzero(initial):
                    key, current = self.data.states[index]
                    self.assertEqual(current["outcomes"], self.lotteries[key[0]]["outcomes"])

    def test_reference_points_follow_the_memory_formulas(self):
        el, fe, c1, stage, z1, z2, zt = self.data.reference_inputs
        delta, delta_le = .5, .25
        expected = {
            "PA": np.where(stage == 1, z1/1.5, np.where(stage == 2, (.5*z1+z2)/1.75, 0)),
            "LE": np.where(stage == 1, el, np.where(stage == 2, (.25*el+c1)/1.25, 0)),
            "FE": fe, "SQ": np.zeros_like(fe)}
        for method in ["tk", "prelec"]:
            spec = ModelSpec("test3", "exp", "sepalpha", "sepmem", method, "fixed_4", 4)
            for rp in ["SQ", "PA", "LE", "FE"]:
                with self.subTest(method=method, rp=rp):
                    layout = build_cluster_layout(spec, rp)
                    theta, _ = expand_free(init_free(layout, np.random.default_rng(1)), layout)
                    theta[9 if method == "prelec" else 8] = delta
                    np.testing.assert_allclose(reference_points(theta, method, delta_le, self.data.reference_inputs),
                                               expected[rp], rtol=1e-15, atol=1e-12)

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
            result = minimize_scalar(objective, bounds=KSI_BOUNDS, method="bounded")
            self.assertAlmostEqual(exact[subject], result.x, delta=1e-5)

    def test_iteration_limit_and_final_likelihood(self):
        spec = ModelSpec("test3", "qh", "sepalpha", "commem", "prelec", "fixed_4", 4)
        result = grid_em_mixture(spec, self.y, self.lotteries, max_iter=1, seed=892,
                                 prepared=self.data)
        self.assertFalse(result["converged"])
        self.assertEqual(result["exit_reason"], "max_iter")
        rows = np.column_stack([norm.logpdf(self.data.observed,
                                            loc=self.scalar_states(theta, "prelec", delta)[self.data.state_index],
                                            scale=result["ksi"][self.data.subject_index]*self.data.spread)
                                for theta, delta in zip(result["thetas"], result["delta_les"])])
        log_l = np.zeros((len(result["subjects"]), 4))
        np.add.at(log_l, self.data.subject_index, rows)
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
