"""Regression checks for warm starts after filtering or reordering participants."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from grid_em import spec_cluster_layouts, _expand_all
from model_spec import ModelSpec, init_free
from run_conditional_grid import initial_values, write_protocol


class WarmStartTests(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec("test2", "exp", "sepalpha", "commem", "prelec",
                              "fixed_pure", 1, "SQ")
        layouts, _ = spec_cluster_layouts(self.spec)
        self.vectors = [init_free(layout, np.random.default_rng(42)) for layout in layouts]
        thetas, deltas = _expand_all(self.vectors, layouts)
        self.result = dict(spec=self.spec, thetas=thetas, delta_les=deltas,
                           pis=np.array([1.]), subjects=["c", "a", "b"],
                           ksi=np.array([.3, .1, .2]))

    def test_subset_and_reordered_participants_keep_their_own_noise(self):
        for subjects, expected in [(["b", "a"], [.2, .1]),
                                    (["a", "b", "c"], [.1, .2, .3])]:
            with self.subTest(subjects=subjects):
                values = initial_values(self.result, self.spec, subjects=subjects)
                np.testing.assert_array_equal(values["ksi"], expected)
                np.testing.assert_array_equal(values["free_vecs"][0], self.vectors[0])
                values["ksi"][0] = 1.
                np.testing.assert_array_equal(self.result["ksi"], [.3, .1, .2])

    def test_default_projection_keeps_full_sample_and_copies_noise(self):
        values = initial_values(self.result, self.spec)
        np.testing.assert_array_equal(values["ksi"], self.result["ksi"])
        self.assertFalse(np.shares_memory(values["ksi"], self.result["ksi"]))

    def test_missing_participant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing 1 target participants"):
            initial_values(self.result, self.spec, subjects=["a", "new"])

    def test_malformed_source_noise_is_rejected(self):
        self.result["ksi"] = np.array([.3, .1])
        with self.assertRaisesRegex(ValueError, "one noise parameter"):
            initial_values(self.result, self.spec, subjects=["a"])


class ProtocolRetryTests(unittest.TestCase):
    def test_failed_preflight_with_empty_restart_directory_can_retry(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            write_protocol(output, {"fingerprint": "old"})
            (output / "test3_example" / "restarts").mkdir(parents=True)
            write_protocol(output, {"fingerprint": "new"})
            self.assertEqual(json.loads((output / "run_protocol.json").read_text()),
                             {"fingerprint": "new"})

    def test_different_protocol_preserves_estimation_files(self):
        for artifact in ["meta.json", "result.pkl", "restarts/seed_42.pkl",
                         "restarts/seed_42.progress.json"]:
            with self.subTest(artifact=artifact), TemporaryDirectory() as directory:
                output = Path(directory)
                write_protocol(output, {"fingerprint": "old"})
                saved = output / "test3_example" / artifact
                saved.parent.mkdir(parents=True)
                saved.write_bytes(b"existing estimation")
                with self.assertRaisesRegex(ValueError, "different protocol"):
                    write_protocol(output, {"fingerprint": "new"})
                self.assertEqual(saved.read_bytes(), b"existing estimation")
                self.assertEqual(json.loads((output / "run_protocol.json").read_text()),
                                 {"fingerprint": "old"})

    def test_matching_protocol_allows_existing_estimates(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            protocol = {"fingerprint": "same"}
            write_protocol(output, protocol)
            saved = output / "test3_example" / "result.pkl"
            saved.parent.mkdir()
            saved.write_bytes(b"existing estimation")
            write_protocol(output, protocol)
            self.assertEqual(saved.read_bytes(), b"existing estimation")


if __name__ == "__main__":
    unittest.main()
