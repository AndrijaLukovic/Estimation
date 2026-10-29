"""Profile constraints, nuisance refits, parallel isolation and checkpoint safety."""
import copy
import csv
import io
import json
import pickle
import unittest
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import functions as f
import grid_em
import profile_lambda as profile
from GlobalSettings import GlobalLottery, GlobalPrelecBounds
from main import get_observed_ce
from model_spec import CANON, ModelSpec, init_free


def make_source(spec, subjects=("a", "b", "c", "d")):
    layouts, identities = grid_em.spec_cluster_layouts(spec)
    rng = np.random.default_rng(57)
    vectors = [init_free(layout, rng) for layout in layouts]
    thetas, deltas = grid_em._expand_all(vectors, layouts)
    return dict(spec=spec, layouts=layouts, rp_identities=identities,
        free_vecs=vectors, thetas=thetas, delta_les=deltas,
        pis=np.ones(spec.C) / spec.C, ksi=np.full(len(subjects), .2),
        subjects=list(subjects), resp=np.ones((len(subjects), spec.C)) / spec.C,
        log_likelihood=-1., optimizer_log=[], valuation_version="conditional_remaining_v1")


class ProfileLayoutTests(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec("test3", "exp", "sepalpha", "commem", "prelec", "fixed_4", 4)
        self.source = make_source(self.spec)

    def test_only_one_component_is_pinned_even_outside_original_bounds(self):
        target, grid = profile.resolve_target(self.source, 2, "lambda", [1., 8.])
        layouts, _ = profile.profile_layouts(self.spec, target, 8.)
        packed = profile.pack_initial(self.source, layouts, target)
        thetas, _ = grid_em._expand_all(packed["free_vecs"], layouts)
        self.assertEqual(thetas[2][3], 8.)
        for j in [0, 1, 3]:
            np.testing.assert_array_equal(thetas[j], self.source["thetas"][j])
            self.assertIn(3, layouts[j].free_positions)
        self.assertNotIn(3, layouts[2].free_positions)
        self.assertEqual(target.reference_bound[1], 8.)
        self.assertIn(target.source_value, grid)

    def test_unrestricted_extended_bound_applies_only_to_focal_component(self):
        target, _ = profile.resolve_target(self.source, 3, "alpha_gain", [5., 8.])
        layouts, _ = profile.profile_layouts(self.spec, target, None)
        for j, layout in enumerate(layouts):
            bounds = profile.layout_bounds(layout, layouts[3], target)
            upper = bounds[layout.free_positions.index(1)][1]
            self.assertEqual(upper, 8. if j == 3 else 5.)

    def test_separate_memory_tail_can_be_fixed_without_fixing_pa_decay(self):
        spec = ModelSpec("test1", "exp", "sepalpha", "sepmem", "prelec", "mixed", 2)
        source = make_source(spec)
        target, _ = profile.resolve_target(source, 0, "delta_le", [.4])
        layouts, _ = profile.profile_layouts(spec, target, .4)
        values = profile.pack_initial(source, layouts, target)
        thetas, deltas = grid_em._expand_all(values["free_vecs"], layouts)
        self.assertEqual(deltas[0], .4)
        self.assertTrue(layouts[1].has_delta_le)
        self.assertIn(CANON["prelec"]["DELTA"], layouts[0].free_positions)
        self.assertEqual(thetas[0][9], source["thetas"][0][9])

    def test_common_memory_aliases_and_linked_curvature(self):
        pa, _ = profile.resolve_target(self.source, 2, "delta_pa")
        le, _ = profile.resolve_target(self.source, 2, "delta_le")
        self.assertEqual(pa, le)
        spec = ModelSpec("test2", "exp", "commonalpha", "commem", "prelec", "fixed_pure", 1, "SQ")
        source = make_source(spec)
        with self.assertRaisesRegex(ValueError, "linked"):
            profile.resolve_target(source, 0, "alpha_loss")
        target, _ = profile.resolve_target(source, 0, "alpha_gain", [2.])
        layouts, _ = profile.profile_layouts(spec, target, 2.)
        packed = profile.pack_initial(source, layouts, target)
        theta, _ = grid_em._expand_all(packed["free_vecs"], layouts)
        self.assertEqual(theta[0][1], 2.)
        self.assertEqual(theta[0][2], 2.)

    def test_fixed_or_invalid_parameters_are_rejected(self):
        for cluster, name in [(3, "delta_pa"), (0, "psi"), (0, "gamma")]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                profile.resolve_target(self.source, cluster, name)
        for name, grid in [("palpha", [0.]), ("alpha_gain", [-1.]), ("palpha", [1.1]), ("r", [float("nan")])]:
            with self.subTest(name=name, grid=grid), self.assertRaises(ValueError):
                profile.resolve_target(self.source, 0, name, grid)

    def test_engine_functions_and_global_bounds_restore_after_exception(self):
        original_layouts, original_bounds = grid_em.spec_cluster_layouts, grid_em.bounds_free
        original_global = copy.deepcopy(GlobalPrelecBounds)
        target, _ = profile.resolve_target(self.source, 1, "palpha")
        layouts, identities = profile.profile_layouts(self.spec, target, .001)
        with self.assertRaisesRegex(RuntimeError, "forced"):
            with profile.constrained_engine(self.spec, layouts, identities, target):
                self.assertIs(grid_em.spec_cluster_layouts(self.spec)[0], layouts)
                raise RuntimeError("forced")
        self.assertIs(grid_em.spec_cluster_layouts, original_layouts)
        self.assertIs(grid_em.bounds_free, original_bounds)
        self.assertEqual(GlobalPrelecBounds, original_global)

    def test_mixed_warm_start_alignment_keeps_noise_and_focal_identity(self):
        spec = ModelSpec("test1", "exp", "sepalpha", "commem", "prelec", "mixed", 2)
        source = make_source(spec)
        source["resp"] = np.array([[.99, .01], [.98, .02], [.03, .97], [.02, .98]])
        source["pis"] = np.array([.4, .6])
        warm = copy.deepcopy(source)
        for name in ["thetas", "delta_les"]:
            warm[name] = warm[name][::-1]
        warm["resp"] = warm["resp"][:, ::-1]
        warm["pis"] = warm["pis"][::-1]
        aligned = profile.align_start(source, warm)
        np.testing.assert_array_equal(aligned["pis"], source["pis"])
        np.testing.assert_array_equal(aligned["thetas"], source["thetas"])
        np.testing.assert_array_equal(aligned["ksi"], source["ksi"])
        target, _ = profile.resolve_target(source, 0, "lambda", [1.])
        diagnostic = profile.point_diagnostics(source, warm, target)
        self.assertTrue(diagnostic["component_switched"])

    def test_job_selection_skips_fixed_parameters_and_deduplicates_aliases(self):
        args = profile.parse_args(["--parameters", "delta_pa", "delta_le", "psi", "--cluster", "LE"])
        jobs, skipped = profile.choose_jobs(self.source, args)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0][0].cluster, 2)
        self.assertEqual(len(skipped), 1)

    def test_separate_parameter_commands_have_separate_default_outputs(self):
        first = profile.resolve_target(self.source, 1, "palpha")
        second = profile.resolve_target(self.source, 3, "alpha_gain")
        path = Path("/tmp/input_run/model/result.pkl")
        self.assertNotEqual(profile.default_output(path, self.source, [first]),
                            profile.default_output(path, self.source, [second]))

    def test_failed_unrestricted_reference_is_not_hidden_by_lr_normalization(self):
        source = copy.deepcopy(self.source)
        source.update(converged=True, exit_reason="ll_tolerance")
        target, _ = profile.resolve_target(source, 0, "lambda", [1.])
        point = copy.deepcopy(source)
        point["log_likelihood"] = 5.
        reference = copy.deepcopy(source)
        reference["log_likelihood"] = 4.
        points = {1.: dict(result=point, forward_ll=5., attempts=[])}
        with TemporaryDirectory() as directory:
            rows = profile.write_tables(Path(directory), source, target, [1.], points, reference)
        self.assertFalse(rows[0]["reference_dominates_grid"])
        self.assertIsNone(rows[0]["lr_statistic"])
        self.assertEqual(rows[0]["delta_ll"], 0.)


class ProfileEstimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        y = get_observed_ce(export_excel=False)
        subjects = sorted(y.participant_label.unique())[:4]
        cls.y = y[y.participant_label.isin(subjects)].copy()
        cls.data = grid_em.prepare_grid_data(cls.y, f.transform(GlobalLottery))

    def source(self, spec):
        source = make_source(spec, self.data.subjects)
        source["n_obs_effective"] = len(self.data.observed)
        source["log_likelihood"], source["resp"] = profile.likelihood(source, self.data)
        return source

    def test_parallel_workers_pin_different_parameters_and_refit_nuisance(self):
        spec = ModelSpec("test3", "exp", "sepalpha", "commem", "prelec", "fixed_4", 4)
        source = self.source(spec)
        targets = [profile.resolve_target(source, 1, "palpha", [.001])[0],
                   profile.resolve_target(source, 3, "alpha_gain", [6.])[0]]
        with ProcessPoolExecutor(max_workers=2, initializer=profile.init_worker, initargs=(self.data,)) as pool:
            futures = [pool.submit(profile.fit_worker, spec, target, value, None, source, 2, 1e-6)
                       for target, value in zip(targets, [.001, 6.])]
            results = [future.result() for future in futures]
        self.assertEqual(results[0]["thetas"][1][5], .001)
        self.assertEqual(results[1]["thetas"][3][1], 6.)
        self.assertNotEqual(results[1]["thetas"][1][5], .001)
        for result in results:
            self.assertFalse(np.array_equal(result["ksi"], source["ksi"]))
            self.assertAlmostEqual(result["pis"].sum(), 1.)
            ll, posterior = profile.likelihood(result, self.data)
            self.assertAlmostEqual(ll, result["log_likelihood"], places=8)
            np.testing.assert_allclose(posterior, result["resp"])

    def test_failed_optimizer_keeps_feasible_point_and_marks_it_unconverged(self):
        spec = ModelSpec("test2", "exp", "sepalpha", "commem", "prelec", "fixed_pure", 1, "SQ")
        source = self.source(spec)
        target, _ = profile.resolve_target(source, 0, "lambda", [4.])
        profile.init_worker(self.data)
        with patch.object(grid_em, "grid_em_mixture", side_effect=FloatingPointError("forced")):
            result = profile.fit_worker(spec, target, 4., None, source, 2, 1e-6)
        self.assertEqual(result["thetas"][0][3], 4.)
        self.assertEqual(result["exit_reason"], "feasible_initial")
        self.assertFalse(result["converged"])

    def test_full_profile_resume_and_protocol_mismatch(self):
        spec = ModelSpec("test2", "exp", "sepalpha", "commem", "prelec", "fixed_pure", 1, "SQ")
        source = self.source(spec)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "inputs" / "model" / "result.pkl"
            path.parent.mkdir(parents=True)
            path.write_bytes(pickle.dumps(source))
            output = root / "profiles"
            argv = ["--result", str(path), "--output-dir", str(output), "--parameters", "lambda",
                    "--grid", "1", "3", "--seeds", "0", "--warm-starts", "0",
                    "--max-iter", "2", "--n-workers", "2"]
            with patch.object(profile, "get_observed_ce", return_value=self.y), redirect_stdout(io.StringIO()):
                profile.main(argv)
            saved = output / "cluster_1_lambda"
            with (saved / "profile.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)  # includes the saved estimate
            summary = json.loads((saved / "profile.json").read_text())
            self.assertTrue(summary["reference_dominates_grid"])
            self.assertTrue(all(row["reverse_done"] for row in [pickle.loads(p.read_bytes()) for p in saved.glob("point_*.pkl")]))
            mtimes = {p: p.stat().st_mtime_ns for p in saved.glob("*.pkl")}
            with patch.object(profile, "get_observed_ce", return_value=self.y), \
                    patch.object(profile, "fit_candidates", side_effect=AssertionError("must resume")), \
                    redirect_stdout(io.StringIO()):
                profile.main(argv)
            self.assertEqual(mtimes, {p: p.stat().st_mtime_ns for p in mtimes})
            changed = argv.copy()
            changed[changed.index("3")] = "4"
            with patch.object(profile, "get_observed_ce", return_value=self.y), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "different protocol"):
                    profile.main(changed)


if __name__ == "__main__":
    unittest.main()
