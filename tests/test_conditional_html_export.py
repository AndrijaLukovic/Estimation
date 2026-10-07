"""Regression checks for the report's scientific transformations."""
import json
from pathlib import Path
import pickle
import unittest

import numpy as np

import export_conditional_html as report


class ConditionalHTMLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=report.ROOT/"results_grid_conditional/test1_qh_sepalpha_commem_prelec_C2/result.pkl"
        if not path.exists():raise unittest.SkipTest("Requires a saved conditional-grid C=2 result")
        with path.open("rb") as fh:cls.result=pickle.load(fh)

    def test_reference_coefficients_are_not_internal_sticks(self):
        clusters=report.cluster_parameters(self.result,1e-4)
        for theta,c in zip(self.result["thetas"],clusters):
            np.testing.assert_allclose(c["weights"],list(report._s_to_a(*theta[6:9]))+[1-sum(report._s_to_a(*theta[6:9]))])
            self.assertAlmostEqual(sum(c["weights"]),1)
        self.assertGreater(abs(clusters[1]["values"]["s2"]-clusters[1]["weights"][1]),.05)

    def test_raw_conversion_keeps_all_subjects_and_handles_nonfinite_logs(self):
        raw=report.readable(self.result)
        self.assertEqual(len(raw["subjects"]),1131)
        self.assertEqual(len(raw["ksi"]),1131)
        self.assertEqual(np.asarray(raw["resp"]).shape,(1131,2))
        self.assertEqual(raw["iter_log"][0]["improvement"],"Infinity")
        json.loads(report.json_text(raw))
        injected={"text":"</script><img src=x onerror=alert(1)>"}
        self.assertNotIn("</script>",report.embed_json(injected))
        self.assertEqual(json.loads(report.embed_json(injected)),injected)

    def test_fixed_parameters_do_not_count_as_boundary_warnings(self):
        clusters=report.cluster_parameters(self.result,1e-4)
        for c in clusters:
            beta=next(p for p in c["parameters"] if p["name"]=="beta")
            self.assertEqual(beta["status"],"固定")
            self.assertIsNone(beta["boundary"])
            self.assertEqual(c["boundary_count"],sum(p["boundary"] is not None for p in c["parameters"]))

    def test_probability_screen_detects_nonmonotone_tk(self):
        self.assertFalse(report.probability_shape("tk",.2)[0])
        self.assertTrue(report.probability_shape("tk",.6)[0])
        self.assertTrue(report.probability_shape("prelec",1e-6)[0])

    def test_shared_scalar_audit_reproduces_saved_likelihood_and_posterior(self):
        data=report.prepare_grid_data(report.get_observed_ce(export_excel=False),report.f.transform(report.GlobalLottery))
        clusters=report.cluster_parameters(self.result,1e-4)
        fit=report.audit_result(self.result,data,clusters)
        self.assertEqual(fit["status"],"verified")
        self.assertLess(abs(fit["ll_gap"]),1e-5)
        self.assertLess(fit["posterior_gap"],1e-7)
        self.assertEqual(len(fit["states"]),40)
        self.assertEqual([row["n"] for row in fit["stage"]],[18096,1130,1131])
        self.assertTrue(all(c["negative_states"]==0 for c in clusters))


if __name__=="__main__":unittest.main()
