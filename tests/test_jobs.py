"""Model selection, config IDs, sample exclusions and the run_grid command line."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from estimation.data import exclusions
from estimation.jobs import generate_jobs, select_jobs, spec_from_config_id
from estimation.spec import ModelSpec, spec_cluster_layouts
from estimation.store import write_protocol
from scripts.run_grid import chosen_jobs, main, parse_args


class SelectJobsTests(unittest.TestCase):
    def test_default_selection_is_the_full_grid_in_estimation_order(self):
        jobs = select_jobs()
        self.assertEqual(sorted(cid for cid, _ in jobs), sorted(cid for cid, _ in generate_jobs()))
        tests = [spec.test for _, spec in jobs]
        self.assertEqual(tests, sorted(tests, key=["test3", "test1", "test2"].index))

    def test_config_ids_round_trip(self):
        for cid, spec in generate_jobs():
            self.assertEqual(spec_from_config_id(cid), spec)
        self.assertEqual(spec_from_config_id("test1_exp_sepalpha_commem_tk_C6").C, 6)

    def test_exponential_utility_config_ids_round_trip(self):
        jobs = select_jobs(utility="exponential")
        self.assertEqual(len(jobs), 144)
        for cid, spec in jobs:
            self.assertEqual(spec.utility, "exponential")
            self.assertIn("_cara", cid)
            self.assertEqual(spec_from_config_id(cid), spec)
        spec = spec_from_config_id("test1_exp_sepalpha_commem_tk_cara_C6")
        self.assertEqual((spec.utility, spec.C), ("exponential", 6))
        self.assertEqual(spec_from_config_id("test3_exp_sepalpha_commem_tk").utility, "power")
        alias = [cid for cid, _ in select_jobs(only=["test2_qh_sepalpha_sepmem_prelec_cara_LE"])]
        self.assertEqual(alias, ["test2_qh_sepalpha_commem_prelec_cara_LE", "test2_qh_sepalpha_sepmem_prelec_cara_LE"])

    def test_test3_with_other_pinned_types_round_trips(self):
        spec = spec_from_config_id("test3_exp_sepalpha_commem_tk_cara_SQ-PA-FE-FE")
        self.assertEqual((spec.rp_set, spec.utility), (("SQ", "PA", "FE", "FE"), "exponential"))
        layouts, identities = spec_cluster_layouts(spec)
        self.assertEqual(identities, ["SQ", "PA", "FE", "FE"])
        usual = spec_cluster_layouts(spec_from_config_id("test3_exp_sepalpha_commem_tk_cara"))[0]
        self.assertEqual([len(layout.free_positions) for layout in layouts],
                         [len(layout.free_positions) for layout in usual[:2]] + [len(usual[3].free_positions)] * 2)
        default = ModelSpec("test3", "exp", "sepalpha", "commem", "tk", "fixed_4", 4, rp_set=("SQ", "PA", "LE", "FE"))
        self.assertEqual(default, spec_from_config_id("test3_exp_sepalpha_commem_tk"))
        with self.assertRaises(ValueError):
            ModelSpec("test1", "exp", "sepalpha", "commem", "tk", "mixed", 4, rp_set=("SQ", "PA", "FE", "FE"))

    def test_malformed_config_ids_are_rejected(self):
        for cid in ["test3_exp_sepalpha_commem_tk_C4", "test1_exp_sepalpha_commem_tk", "test2_exp_sepalpha_commem_tk_XX",
                    "test1_exp_sepalpha_commem_tk_C0", "test4_exp_sepalpha_commem_tk", "test3_exp_wrong_commem_tk",
                    "test3_exp_sepalpha_commem_tk_cara_cara", "test3_exp_sepalpha_commem_cara_tk",
                    "test1_exp_sepalpha_commem_tk_C4_cara", "test3_exp_sepalpha_commem_tk_SQ-PA-LE-FE",
                    "test3_exp_sepalpha_commem_tk_SQ-PA-XX-FE", "test3_exp_sepalpha_commem_tk_SQ-PA-FE"]:
            with self.subTest(cid=cid), self.assertRaises(ValueError):
                spec_from_config_id(cid)

    def test_filters_pick_one_model(self):
        jobs = select_jobs(tests=["test3"], discounting=["exp"], curvature=["sepalpha"], memory=["commem"], pw=["tk"])
        self.assertEqual([cid for cid, _ in jobs], ["test3_exp_sepalpha_commem_tk"])

    def test_test1_classes_ascend_after_test3(self):
        jobs = select_jobs(tests=["test1", "test3"], discounting=["exp"], curvature=["sepalpha"],
                           memory=["commem"], pw=["tk"], clusters=[6, 4, 5])
        self.assertEqual([cid for cid, _ in jobs], ["test3_exp_sepalpha_commem_tk"] +
                         [f"test1_exp_sepalpha_commem_tk_C{c}" for c in (4, 5, 6)])

    def test_pure_reference_sepmem_brings_its_commem_source_first(self):
        jobs = [cid for cid, _ in select_jobs(only=["test2_qh_sepalpha_sepmem_prelec_LE"])]
        self.assertEqual(jobs, ["test2_qh_sepalpha_commem_prelec_LE", "test2_qh_sepalpha_sepmem_prelec_LE"])
        mixed = [cid for cid, _ in select_jobs(only=["test1_qh_sepalpha_sepmem_prelec_C2"])]
        self.assertEqual(mixed, ["test1_qh_sepalpha_sepmem_prelec_C2"])


class ExclusionTests(unittest.TestCase):
    def setUp(self):
        self.observed = pd.DataFrame(dict(participant_label=["a", "a", "b", "c"]))

    def test_no_rule_excludes_nobody(self):
        table = exclusions(self.observed)
        self.assertTrue(table.empty)
        self.assertEqual(table.in_sample.dtype, bool)

    def test_listed_participants_are_excluded_and_unknown_labels_flagged(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "drop.csv"
            pd.DataFrame(dict(participant_label=[" b", "zz", "b"])).to_csv(path, index=False)
            table = exclusions(self.observed, files=[path])
        self.assertEqual(table.participant_label.tolist(), ["b", "zz"])
        self.assertEqual(table.in_sample.tolist(), [True, False])
        self.assertEqual(table.reason.tolist(), ["listed in drop.csv"] * 2)


class CommandLineTests(unittest.TestCase):
    def test_only_and_filters_are_exclusive(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--only", "test3_exp_sepalpha_commem_tk", "--pw", "tk"])

    def test_utility_option_selects_cara_models_and_excludes_only(self):
        jobs = chosen_jobs(parse_args(["--tests", "test3", "--pw", "tk", "--utility", "exponential", "--list"]))
        self.assertEqual(len(jobs), 8)
        self.assertTrue(all(spec.utility == "exponential" and cid.endswith("_tk_cara") for cid, spec in jobs))
        self.assertTrue(all(spec.utility == "power" for _, spec in chosen_jobs(parse_args(["--list"]))))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--only", "test3_exp_sepalpha_commem_tk", "--utility", "exponential"])
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            main(["--only", "test3_exp_sepalpha_commem_tk", "test3_exp_sepalpha_commem_tk_cara", "--list"])

    def test_filters_default_to_everything_not_given(self):
        jobs = chosen_jobs(parse_args(["--tests", "test2", "--pw", "tk", "--rp", "FE", "--list"]))
        self.assertEqual(len(jobs), 8)
        self.assertTrue(all(spec.rp_identity == "FE" and spec.pw == "tk" for _, spec in jobs))

    def test_output_directory_is_required_except_for_listing(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--tests", "test3"])
        self.assertIsNone(parse_args(["--list"]).output_dir)

    def test_invalid_shares_and_classes_are_rejected(self):
        for argv in (["--exclude-endpoint", "0"], ["--exclude-endpoint", "1.5"], ["--clusters", "0"], ["--seeds", "0"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)


class ProtocolMessageTests(unittest.TestCase):
    def write_existing(self, directory, protocol):
        output = Path(directory)
        write_protocol(output, protocol)
        (output / "test3_example").mkdir()
        (output / "test3_example" / "result.pkl").write_bytes(b"saved")
        return output

    def test_message_names_directory_and_changed_inputs(self):
        old = dict(inputs=dict(sample_sha256="a", bounds=1), seeds=[1], fingerprint="old")
        new = dict(inputs=dict(sample_sha256="b", bounds=1), seeds=[2], fingerprint="new")
        with TemporaryDirectory() as directory:
            output = self.write_existing(directory, old)
            with self.assertRaisesRegex(ValueError, r"differs in: inputs\.sample_sha256, seeds"):
                write_protocol(output, new)

    def test_message_flags_pre_refactor_directories(self):
        with TemporaryDirectory() as directory:
            output = self.write_existing(directory, dict(source_hashes={}, fingerprint="old"))
            with self.assertRaisesRegex(ValueError, "before the refactor"):
                write_protocol(output, dict(inputs={}, fingerprint="new"))


if __name__ == "__main__":
    unittest.main()
