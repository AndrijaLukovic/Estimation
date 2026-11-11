"""Random-start design: seeds, CARA alpha draws, short-run screening, reuse of
finished starts, replication counts and warm starts from several C-1 fits."""
import contextlib
import io
import json
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from estimation.config import LOTTERIES, SEEDS, random_seeds
from estimation.data import observed_ce
from estimation.em import grid_em_mixture
from estimation.jobs import spec_from_config_id
from estimation.lottery_paths import transform
from estimation.multistart import _restart_worker, grid_em_multistart, ll_after
from estimation.prepare import prepare_grid_data
from estimation.run_summary import restart_hits
from estimation.spec import LOGUNIF_ALPHA_MIN, expand_all, init_free, spec_cluster_layouts
from estimation.store import load_pickle
from estimation.warm_starts import drop_duplicate_starts, job_initializations

from scripts.run_grid import check_reuse, parse_args

CARA_C2 = "test1_exp_sepalpha_commem_tk_cara_C2"


def small_sample(n=4):
    y = observed_ce()
    return y[y.participant_label.isin(sorted(y.participant_label.unique())[:n])]


class SeedTests(unittest.TestCase):
    def test_more_seeds_extend_the_original_hundred_without_repeats(self):
        self.assertEqual(random_seeds(100), SEEDS)
        many = random_seeds(1000)
        self.assertEqual(len(set(many)), 1000)
        self.assertEqual(random_seeds(300)[:200], random_seeds(200))
        with self.assertRaises(ValueError):
            random_seeds(0)


class AlphaDrawTests(unittest.TestCase):
    def setUp(self):
        self.layout = spec_cluster_layouts(spec_from_config_id(CARA_C2))[0][0]
        self.alpha = [self.layout.free_positions.index(position) for position in (1, 2)]

    def test_default_draw_is_unchanged_by_the_option(self):
        np.testing.assert_array_equal(init_free(self.layout, np.random.default_rng(5)),
                                      init_free(self.layout, np.random.default_rng(5), alpha_init="normal"))

    def test_log_uniform_draw_spans_the_decades_inside_the_bounds(self):
        rng = np.random.default_rng(0)
        alphas = np.array([init_free(self.layout, rng, alpha_init="logunif")[self.alpha] for _ in range(2000)]).ravel()
        self.assertTrue(np.all(np.abs(alphas) <= 0.3) and np.all(np.abs(alphas) >= LOGUNIF_ALPHA_MIN))
        self.assertAlmostEqual(np.mean(alphas > 0), 0.5, delta=0.05)
        share_large = np.mean(np.abs(alphas) > 0.03)       # log10(0.3/0.03) / log10(0.3/1e-4) = 0.29
        self.assertAlmostEqual(share_large, 1 / np.log10(3000), delta=0.04)

    def test_power_utility_has_no_log_uniform_alpha(self):
        layout = spec_cluster_layouts(spec_from_config_id("test1_exp_sepalpha_commem_tk_C2"))[0][0]
        with self.assertRaisesRegex(ValueError, "exponential"):
            init_free(layout, np.random.default_rng(0), alpha_init="logunif")


class ScreeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = spec_from_config_id(CARA_C2)
        cls.y = small_sample()
        cls.prepared = prepare_grid_data(cls.y, transform(LOTTERIES))

    def task(self, seed, max_iter, previous=None, alpha_init="normal"):
        return dict(spec=self.spec, seed=seed, max_iter=max_iter, tol=0.0, initial_values=None,
                    prepared=self.prepared, bounds=None, alpha_init=alpha_init, progress_path=None,
                    restart_id=f"seed_{seed}", checkpoint=None, previous=previous)

    def test_continuing_a_screening_run_equals_one_uninterrupted_run(self):
        whole = grid_em_mixture(self.spec, max_iter=6, tol=0.0, seed=7, prepared=self.prepared)
        first = _restart_worker(self.task(7, 2))
        joined = _restart_worker(self.task(7, 6, previous=first))
        self.assertEqual(joined["n_iter"], 6)
        self.assertEqual(ll_after(whole, 2), first["log_likelihood"])
        np.testing.assert_allclose([row["log_likelihood"] for row in joined["iter_log"]],
                                   [row["log_likelihood"] for row in whole["iter_log"]], rtol=0, atol=1e-8)
        self.assertEqual([row["iteration"] for row in joined["iter_log"]], list(range(1, 7)))
        np.testing.assert_allclose(joined["log_likelihood"], whole["log_likelihood"], rtol=0, atol=1e-8)
        np.testing.assert_allclose(np.concatenate(joined["thetas"]), np.concatenate(whole["thetas"]), atol=1e-6)

    def test_screening_continues_the_best_share_and_resumes_from_checkpoints(self):
        with TemporaryDirectory() as directory:
            run = dict(y=self.y, max_iter=5, tol=0.0, n_workers=2, verbose=False, checkpoint_dir=Path(directory),
                       screen_iter=2, screen_keep=0.5, logunif_seeds=[3, 4])
            best = grid_em_multistart(self.spec, [1, 2, 3, 4], **run)
            rows = {row["restart_id"]: row for row in best["restart_summaries"]}
            self.assertEqual(set(rows), {"seed_1", "seed_2", "seed_3_logunif", "seed_4_logunif"})
            screen = {label: load_pickle(Path(directory) / "screen" / f"{label}.pkl")["log_likelihood"]
                      for label in rows}
            top = set(sorted(screen, key=screen.get, reverse=True)[:2])
            self.assertEqual({label for label, row in rows.items() if row["stage"] == "continued"}, top)
            self.assertEqual(best["n_restarts_ok"], 2)
            self.assertEqual(best["n_screened_out"], 2)
            self.assertEqual(best["screen"], dict(iter=2, keep=0.5, n_kept=2))
            for label in top:
                self.assertEqual(rows[label]["n_iter"], 5)
                seed = int(label.split("_")[1])
                alpha_init = "logunif" if "logunif" in label else "normal"
                whole = grid_em_mixture(self.spec, max_iter=5, tol=0.0, seed=seed, prepared=self.prepared,
                                        alpha_init=alpha_init)
                self.assertAlmostEqual(rows[label]["log_likelihood"], whole["log_likelihood"], places=7)
            files = sorted(path.name for path in Path(directory).rglob("*.pkl"))
            again = grid_em_multistart(self.spec, [1, 2, 3, 4], **run)
            self.assertEqual(again["log_likelihood"], best["log_likelihood"])
            self.assertEqual(sorted(path.name for path in Path(directory).rglob("*.pkl")), files)

    def test_finished_random_starts_of_another_run_are_copied(self):
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            run = dict(y=self.y, max_iter=3, tol=0.0, n_workers=1, verbose=False)
            old = grid_em_multistart(self.spec, [1], checkpoint_dir=Path(first), **run)
            new = grid_em_multistart(self.spec, [1, 2], checkpoint_dir=Path(second), reuse_dir=Path(first), **run)
            copied = load_pickle(Path(second) / "seed_1.pkl")
            self.assertEqual(copied["copied_from"], str(Path(first) / "seed_1.pkl"))
            self.assertEqual(copied["log_likelihood"], old["log_likelihood"])
            self.assertNotIn("copied_from", load_pickle(Path(second) / "seed_2.pkl"))
            self.assertEqual(new["n_restarts_ok"], 2)


class HitsTests(unittest.TestCase):
    def test_hits_count_finished_random_starts_only(self):
        rows = [dict(seed=None, log_likelihood=-10.0), dict(seed=1, log_likelihood=-10.005),
                dict(seed=2, log_likelihood=-10.5), dict(seed=3, log_likelihood=-10.0, stage="screened_out"),
                dict(seed=4, log_likelihood=-10.05, stage="continued")]
        result = dict(log_likelihood=-10.0, restart_summaries=rows)
        self.assertEqual(restart_hits(result), (1, 3))
        self.assertEqual(restart_hits(result, 0.1), (2, 3))


class SplitSourceTests(unittest.TestCase):
    def fit(self, rng_seed):
        spec = spec_from_config_id("test1_exp_sepalpha_commem_tk_cara_C1")
        layouts, identities = spec_cluster_layouts(spec)
        vectors = [init_free(layout, np.random.default_rng(rng_seed)) for layout in layouts]
        thetas, deltas = expand_all(vectors, layouts)
        return dict(spec=spec, thetas=thetas, delta_les=deltas, pis=np.array([1.]), subjects=["a", "b"],
                    ksi=np.array([.2, .3]), rp_identities=identities)

    def test_every_warm_directory_is_split_and_repeats_are_dropped(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name, rng_seed in (("w1", 1), ("w2", 1), ("w3", 2)):
                (root / name / "test1_exp_sepalpha_commem_tk_cara_C1").mkdir(parents=True)
                with open(root / name / "test1_exp_sepalpha_commem_tk_cara_C1" / "result.pkl", "wb") as handle:
                    pickle.dump(self.fit(rng_seed), handle)
            (root / "out").mkdir()
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                starts = job_initializations(spec_from_config_id(CARA_C2), ["a", "b"], root / "out", "fp",
                                             warm_dirs=[root / "w1", root / "w2", root / "w3"])
            self.assertEqual(set(starts), {"split_C1_class1", "split_C1_w3_class1"})
            self.assertIn("split_C1_w2_class1: identical to split_C1_class1", printed.getvalue())

    def test_identical_starts_are_dropped_in_order(self):
        start = dict(free_vecs=[np.ones(3)], pis=np.ones(1), ksi=np.ones(2))
        other = dict(free_vecs=[np.zeros(3)], pis=np.ones(1), ksi=np.ones(2))
        with contextlib.redirect_stdout(io.StringIO()):
            kept = drop_duplicate_starts(dict(a=start, b=dict(start), c=other))
        self.assertEqual(list(kept), ["a", "c"])


class OptionTests(unittest.TestCase):
    def test_screening_options_go_together_and_stay_in_range(self):
        for argv in (["--screen-iter", "20"], ["--screen-keep", ".3"],
                     ["--screen-iter", "0", "--screen-keep", ".3"], ["--screen-iter", "20", "--screen-keep", "0"],
                     ["--seeds", "10", "--logunif-alpha", "11"], ["--seeds", "100000"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                parse_args(argv + ["--output-dir", "x"])
        args = parse_args(["--seeds", "500", "--logunif-alpha", "250", "--screen-iter", "20", "--screen-keep", ".3",
                           "--output-dir", "x"])
        self.assertEqual((args.seeds, args.logunif_alpha, args.screen_iter, args.screen_keep), (500, 250, 20, .3))

    def test_restarts_are_reused_only_under_the_same_inputs_and_tolerances(self):
        with TemporaryDirectory() as directory:
            old = dict(inputs=dict(sample_sha256="s"), max_iter=1000, tol=1e-6, warm_dirs=["a"], fingerprint="f")
            (Path(directory) / "run_protocol.json").write_text(json.dumps(old))
            check_reuse(Path(directory), dict(old, warm_dirs=["b"], seeds=[1, 2], fingerprint="g"))
            with self.assertRaisesRegex(ValueError, "tol"):
                check_reuse(Path(directory), dict(old, tol=1e-8))
            with self.assertRaisesRegex(ValueError, "inputs.sample_sha256"):
                check_reuse(Path(directory), dict(old, inputs=dict(sample_sha256="t")))


if __name__ == "__main__":
    unittest.main()
