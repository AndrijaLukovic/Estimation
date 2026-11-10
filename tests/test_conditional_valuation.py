"""Numerical regression checks for conditional remaining-payoff estimation."""
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import norm

from estimation import cpt
from estimation.config import CARA_BOUNDS, KSI_BOUNDS, LOTTERIES
from estimation.data import observed_ce
from estimation.em import grid_em_mixture
from estimation.likelihood import noise_mstep, result_loglik
from estimation.lottery_paths import conditional_lottery, transform
from estimation.multistart import grid_em_multistart
from estimation.prepare import prepare_grid_data
from estimation.spec import (CANON, ModelSpec, all_specs_16, bounds_free, build_cluster_layout, default_bounds,
                             init_free, expand_free)
from estimation.valuation import named_parameters, reference_points, scalar_ce
from estimation.warm_starts import equivalent_result


class ConditionalValuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lotteries = transform(LOTTERIES)
        cls.y = observed_ce()
        cls.data = prepare_grid_data(cls.y, cls.lotteries)

    def scalar_states(self, theta, method, delta_le, utility="power"):
        """State CEs from the scalar CPT reference implementation."""
        values = named_parameters(theta, method)
        references = reference_points(theta, method, delta_le, self.data.reference_inputs)
        return np.array([scalar_ce(values, float(reference), lottery, method, utility)[0]
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

    def test_exponential_utility_is_cara_around_the_reference_point(self):
        exp = dict(utility="exponential", lamb=2.25)
        for R in [-40., 0., 13.]:
            self.assertEqual(cpt.u(R, R, .01, alpha_loss=.005, **exp), 0)
            # Slopes 1 and lambda at R, so lambda keeps its loss-aversion meaning.
            h = 1e-7
            self.assertAlmostEqual(cpt.u(R+h, R, .01, alpha_loss=.005, **exp)/h, 1, places=6)
            self.assertAlmostEqual(-cpt.u(R-h, R, .01, alpha_loss=.005, **exp)/h, 2.25, places=6)
        # alpha > 0: concave gains, convex losses, bounded by 1/alpha and -lambda/alpha_loss.
        gains = [cpt.u(x, 0, .01, alpha_loss=.02, **exp) for x in (100, 200, 300)]
        losses = [cpt.u(-x, 0, .01, alpha_loss=.02, **exp) for x in (100, 200, 300)]
        self.assertLess(gains[2]-gains[1], gains[1]-gains[0])
        self.assertTrue(losses[0] > losses[1] > losses[2])
        self.assertGreater(losses[0]-losses[1], losses[1]-losses[2])
        self.assertLess(max(gains), 100)
        self.assertGreater(min(losses), -2.25/.02)
        # alpha < 0 is convex in gains; alpha = 0 is linear.
        self.assertGreater(cpt.u(200, 0, -.01, **exp) - cpt.u(100, 0, -.01, **exp), cpt.u(100, 0, -.01, **exp))
        self.assertEqual(cpt.u(37., 0, 0., alpha_loss=0., **exp), 37.)
        self.assertEqual(cpt.u(-37., 0, 0., alpha_loss=0., **exp), -2.25*37.)
        # Exact inverse across signs and the full payoff range at the bound.
        low, high = CARA_BOUNDS
        for alpha in [low, -1e-9, 0., 1e-9, .003, high]:
            for alpha_loss in [low, 0., .004, high]:
                for x in [-2155., -12.5, -1e-6, 0., 1e-6, 7., 2155.]:
                    # Beyond ~37 a concave u has saturated and has no inverse (see cara_ce).
                    if (alpha if x >= 3 else alpha_loss) * abs(x - 3) > 30:
                        continue
                    y = cpt.u(x, 3., alpha, alpha_loss=alpha_loss, **exp)
                    self.assertAlmostEqual(cpt.u_inv(y, 3., alpha, alpha_loss=alpha_loss, **exp), x,
                                           delta=1e-6 * max(1, abs(x)))
        with self.assertRaises(ValueError):
            cpt.u(1, utility="expo")
        with self.assertRaises(ValueError):
            cpt.u_inv(1, utility="expo")

    def test_exponential_utility_bounds_and_initial_values(self):
        power, cara = default_bounds("power"), default_bounds("exponential")
        for method, ix in CANON.items():
            for key, position in ix.items():
                if key != "LEN":
                    expected = CARA_BOUNDS if key in ("AG", "AL") else power[method][position]
                    self.assertEqual(tuple(cara[method][position]), tuple(expected))
            spec = ModelSpec("test1", "exp", "sepalpha", "sepmem", method, "mixed", 2, utility="exponential")
            layout = build_cluster_layout(spec)
            self.assertEqual(layout.utility, "exponential")
            rng = np.random.default_rng(3)
            for _ in range(50):
                vector = init_free(layout, rng)
                low, high = np.array(bounds_free(layout)).T
                self.assertTrue(np.all((vector >= low) & (vector <= high)))

    def test_vectorized_exponential_prediction_matches_scalar_formula(self):
        rng = np.random.default_rng(782)
        low, high = CARA_BOUNDS
        for disc, curvature, memory, method in all_specs_16():
            for rp in ["SQ", "PA", "LE", "FE", None]:
                spec = ModelSpec("test3" if rp else "test1", disc, curvature, memory,
                                 method, "fixed_4" if rp else "mixed", 4, utility="exponential")
                layout = build_cluster_layout(spec, rp)
                theta, delta = expand_free(init_free(layout, rng), layout)
                for curvature_values in [None, [high, low, 6], [low, high, 6], [high, high, 6], [0., 0., 1.]]:
                    with self.subTest(spec=spec.config_id(), rp=rp, curvature=curvature_values):
                        current = theta.copy()
                        if curvature_values is not None:
                            current[1:4] = curvature_values
                        slow = self.scalar_states(current, method, delta, "exponential")
                        fast = self.data.predict(current, method, delta, "exponential")
                        self.assertTrue(np.isfinite(fast).all())
                        np.testing.assert_allclose(fast, slow, rtol=2e-9, atol=2e-8)
        # Risk neutrality (alpha = 0, lambda = 1, no probability weighting) prices
        # every state at its discounted expected value under either utility.
        spec = ModelSpec("test3", "exp", "sepalpha", "commem", "tk", "fixed_4", 4, utility="exponential")
        layout = build_cluster_layout(spec, "SQ")
        theta, delta = expand_free(init_free(layout, rng), layout)
        theta[1:5] = [0., 0., 1., 1.]
        theta[0] = 0.
        power = theta.copy()
        power[1:3] = 1.
        np.testing.assert_allclose(self.data.predict(theta, "tk", delta, "exponential"),
                                   self.data.predict(power, "tk", delta), rtol=1e-12, atol=1e-9)

    def test_exponential_ce_is_exact_where_utility_saturates(self):
        a, gamma, lamb = CARA_BOUNDS[1], .6, 2.
        w = cpt.pw(.5, gamma)
        lottery = lambda x, y: {"name": "l", "outcomes": {"good": {.5: [x]}, "bad": {.5: [y]}}}
        ce = lambda x, y: cpt.cara_ce(lottery(x, y), 0., 0., a, a, lamb, gamma=gamma)
        # Gains: u is 1/a in floats at both outcomes, yet the CE is still defined.
        self.assertEqual(cpt.u(2000., 0, a, alpha_loss=a, lamb=lamb, utility="exponential"),
                         cpt.u(500., 0, a, alpha_loss=a, lamb=lamb, utility="exponential"))
        self.assertAlmostEqual(ce(2000., 500.), -math.log(w * math.exp(-a * 2000) + (1 - w) * math.exp(-a * 500)) / a,
                               delta=1e-10)
        self.assertAlmostEqual(ce(2000., 500.), 500 - math.log(1 - w) / a, delta=1e-9)
        # Losses mirror gains.
        self.assertAlmostEqual(ce(-500., -2000.), -500 + math.log(1 - w) / a, delta=1e-9)
        # A degenerate lottery is priced at its outcome at any curvature.
        certain = {"name": "l", "outcomes": {"only": {1.: [2000.]}}}
        self.assertAlmostEqual(cpt.cara_ce(certain, 0., 0., a, a, lamb, gamma=gamma), 2000., delta=1e-9)
        # Mixed, value >= 0: the loss enters as a * |V_loss|.
        gain, loss, up, down = 500., 1., cpt.pw(.95, gamma), cpt.pw(.05, gamma)
        mixed = {"name": "l", "outcomes": {"good": {.95: [gain]}, "bad": {.05: [-loss]}}}
        value = up * cpt.cara(gain, a) - lamb * down * cpt.cara(loss, a)
        self.assertGreater(a * value, .5)
        kept = 1 - up + up * math.exp(-a * gain) + a * lamb * down * cpt.cara(loss, a)
        self.assertAlmostEqual(cpt.cara_ce(mixed, 0., 0., a, a, lamb, gamma=gamma), -math.log(kept) / a, delta=1e-9)

    def test_risk_premium_reads_curvature_as_risk_attitude(self):
        # Power: closed form; alpha < 1 is risk averse over gains and risk seeking over losses.
        for alpha in (.5, 1., 1.8):
            gain, loss = cpt.risk_premium(100, alpha, alpha)
            self.assertAlmostEqual(gain, 1 - 2 ** (1 - 1 / alpha), places=12)
            self.assertAlmostEqual(loss, 2 ** (1 - 1 / alpha) - 1, places=12)
        # CARA: closed form over the whole bound range; alpha > 0 is risk averse over
        # gains and risk seeking over losses, alpha = 0 neutral.
        low, high = CARA_BOUNDS
        for alpha in (low, -.01, 0., .0825, high):
            for stake in (20, 100, 500):
                with self.subTest(alpha=alpha, stake=stake):
                    half = stake / 2 if alpha == 0 else -math.log(.5 + .5 * math.exp(-alpha * stake)) / alpha
                    gain, loss = cpt.risk_premium(stake, alpha, alpha, "exponential")
                    self.assertAlmostEqual(gain, 1 - 2 * half / stake, places=12)
                    self.assertAlmostEqual(loss, 2 * half / stake - 1, places=12)
                    self.assertTrue(-1 < gain < 1 and -1 < loss < 1)
                    self.assertEqual((np.sign(gain), np.sign(loss)), (np.sign(alpha), -np.sign(alpha)))
        # Each side uses its own curvature.
        self.assertEqual(cpt.risk_premium(100, .01, -.02, "exponential")[0], cpt.risk_premium(100, .01, .5, "exponential")[0])
        self.assertEqual(cpt.risk_premium(100, .01, -.02, "exponential")[1], cpt.risk_premium(100, .7, -.02, "exponential")[1])

    def test_exponential_utility_estimation_runs_within_bounds(self):
        spec = ModelSpec("test2", "exp", "sepalpha", "commem", "tk", "fixed_pure", 1, "FE", utility="exponential")
        result = grid_em_mixture(spec, max_iter=3, seed=5, prepared=self.data)
        self.assertTrue(np.isfinite(result["log_likelihood"]))
        ll, _ = result_loglik(result, self.data)
        self.assertAlmostEqual(ll, result["log_likelihood"], places=8)
        low, high = CARA_BOUNDS
        for position in (CANON["tk"]["AG"], CANON["tk"]["AL"]):
            self.assertTrue(low <= result["thetas"][0][position] <= high)
        power = ModelSpec("test2", "exp", "sepalpha", "commem", "tk", "fixed_pure", 1, "FE")
        with self.assertRaises(ValueError):
            grid_em_mixture(spec, max_iter=1, seed=5, prepared=self.data,
                            layouts=[build_cluster_layout(power, "FE")])

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
