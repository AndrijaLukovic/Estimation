"""summary.txt: protocol formats, loss aversion, saved runs and the run_grid hook."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from estimation.config import BOUNDS
from estimation.run_summary import find_runs, loss_aversion, run_bounds, run_summary
from estimation.spec import CANON, default_bounds
from scripts.run_grid import main as run_grid
from scripts.summarize_run import main as summarize, resolve


class ProtocolTests(unittest.TestCase):
    def test_bounds_of_every_protocol_format(self):
        tk = CANON["tk"]
        self.assertEqual(run_bounds({}, "tk"), list(BOUNDS["tk"]))       # legacy grid: no protocol
        old = run_bounds({"lambda_bound": [1e-6, 20], "weight_bound": [0.28, 1]}, "tk")
        self.assertEqual((old[tk["LAM"]], old[tk["GAM"]]), ((1e-6, 20), (0.28, 1)))
        self.assertEqual(run_bounds({"weight_bound": [0.28, 1]}, "prelec"), list(BOUNDS["prelec"]))
        table = [list(bound) for bound in default_bounds("exponential")["tk"]]
        new = run_bounds({"inputs": {"bounds": {"tk": table}}}, "tk")
        self.assertEqual(new[tk["AG"]], (-0.3, 0.3))


class LossAversionTests(unittest.TestCase):
    def test_power_utility_matches_the_closed_form(self):
        for stake in (5, 20, 50):
            self.assertAlmostEqual(loss_aversion(1.2, 0.48, 20, stake), 20 * stake ** (0.48 - 1.2))

    def test_cara_tends_to_lambda_at_small_stakes(self):
        self.assertAlmostEqual(loss_aversion(0.01, -0.002, 1.5, 1e-6, "exponential"), 1.5, places=6)


class SavedRunTests(unittest.TestCase):
    def test_every_saved_run_lists_all_its_finished_models(self):
        runs = find_runs()
        if not runs:
            self.skipTest("Requires saved runs under results/")
        for run in runs:
            with self.subTest(run=run.name):
                text = run_summary(run)
                for path in run.glob("test*/meta.json"):
                    meta = json.loads(path.read_text())
                    if meta.get("status") == "done":
                        self.assertIn("\n" + meta["config_id"] + "\n", text)

    def test_runs_resolve_by_name_and_unknown_names_fail(self):
        runs = find_runs()
        if not runs:
            self.skipTest("Requires saved runs under results/")
        unique = [run for run in runs if [other.name for other in runs].count(run.name) == 1]
        self.assertEqual(resolve(unique[0].name), unique[0])
        with self.assertRaises(SystemExit):
            resolve("no_such_run")


class RunGridHookTests(unittest.TestCase):
    def test_run_grid_writes_summary_txt_and_the_script_rewrites_it(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            run_grid(["--only", "test2_exp_sepalpha_commem_tk_SQ", "--seeds", "1", "--max-iter", "1",
                      "--n-workers", "1", "--output-dir", str(output)])
            written = (output / "summary.txt").read_text()
            self.assertIn("test2_exp_sepalpha_commem_tk_SQ", written)
            target = Path(directory) / "copy.txt"
            summarize([str(output), "--output", str(target)])
            self.assertEqual(target.read_text().split("\n", 2)[2], written.split("\n", 2)[2])


if __name__ == "__main__":
    unittest.main()
