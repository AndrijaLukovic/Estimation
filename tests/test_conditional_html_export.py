"""Regression checks for the report's scientific transformations."""
from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import pickle
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs

import numpy as np

import export_conditional_html as report
from model_spec import ModelSpec, build_cluster_layout, expand_free, init_free
from report_storage import read_bundle_value


class ReportLanguageTests(unittest.TestCase):
    def test_cli_language_defaults_and_output_override(self):
        chinese=report.parse_args([])
        english=report.parse_args(["--language","en"])
        self.assertEqual(chinese.language,"zh")
        self.assertEqual(chinese.output_dir,report.ROOT/"reports/grid_em_conditional_explorer")
        self.assertEqual(english.output_dir,report.ROOT/"reports/grid_em_conditional_explorer_en")
        self.assertNotEqual(chinese.output_dir,english.output_dir)
        custom=report.parse_args(["--lang","en","--output-dir","reports/custom"])
        self.assertEqual(custom.output_dir,Path("reports/custom"))
        with patch("sys.stderr",new=io.StringIO()):
            with self.assertRaises(SystemExit):report.parse_args(["--language","fr"])
            with self.assertRaises(SystemExit):report.parse_args(["--language","en","--boundary-tol","-1"])

    def test_english_templates_and_script_have_no_untranslated_chinese(self):
        for name in ["dashboard.html","detail.html","app.js"]:
            with self.subTest(asset=name):
                source=(report.ASSETS/name).read_text(encoding="utf-8")
                self.assertRegex(source,r"[\u3400-\u9fff]")
                self.assertEqual(report.localize(source,"zh"),source)
                self.assertNotRegex(report.localize(source,"en"),r"[\u3400-\u9fff]")

    @staticmethod
    def make_result():
        spec=ModelSpec("test1","exp","commonalpha","commem","prelec","mixed",1)
        layout=build_cluster_layout(spec)
        free=init_free(layout,np.random.default_rng(42))
        theta,delta_le=expand_free(free,layout)
        theta[report.CANON[spec.pw]["R"]]=report.bounds_free(layout)[0][0]
        theta[report.CANON[spec.pw]["S1"]]=1
        return dict(spec=spec,thetas=[theta],delta_les=[delta_le],layouts=[layout],free_vecs=[free],
            resp=np.ones((2,1)),ksi=np.array([.25,.5]),subjects=["参与者甲","person2"],
            pis=np.ones(1),rp_identities=[None],log_likelihood=-3.,n_obs=4,
            converged=False,n_iter=1,exit_reason="max_iter",valuation_version="conditional_remaining_v1",
            original_note="自由估计；原始中文",iter_log=[],optimizer_log=[])

    @staticmethod
    def read_payload(path):
        location,separator,query=str(path).partition("?")
        if separator:
            root=Path(location).parent;key=parse_qs(query)["id"][0]
            metadata=json.loads((root/"catalog.json").read_text())
            payload=read_bundle_value(root,key,metadata["storage"]["details"][key])
            payload["summary"].update(metadata["storage"]["details"][key].get("summary_extra",{}))
            return payload
        html=path.read_text(encoding="utf-8")
        return json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>',html).group(1))

    def test_bilingual_exports_keep_numbers_and_raw_data_and_localize_diagnostics(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);inputs=root/"inputs";model=inputs/"model"
            (model/"restarts").mkdir(parents=True)
            result=self.make_result()
            original=pickle.dumps(result)
            (model/"result.pkl").write_bytes(original)
            (model/"meta.json").write_text(json.dumps({"note":"自由估计"}),encoding="utf-8")
            restart={**result,"restart_id":"seed_123","log_likelihood":-4.}
            (model/"restarts/seed_123.pkl").write_bytes(pickle.dumps(restart))
            (inputs/"other.pkl").write_bytes(pickle.dumps({"note":"自由估计"}))
            (inputs/"broken.pkl").write_bytes(b"invalid pickle")
            catalogs={}
            for language in ["zh","en"]:
                output=root/language
                args=report.parse_args(["--language",language,"--input-dir",str(inputs),
                    "--output-dir",str(output),"--audit","none"])
                with redirect_stdout(io.StringIO()):catalogs[language]=report.export(args)
                self.assertIn(f'lang="{"en" if language=="en" else "zh-CN"}"',(output/"index.html").read_text())
                self.assertEqual(json.loads((output/"manifest.json").read_text())["language"],language)
                for figure in ["model_bic_overview","restart_ll_spread"]:
                    self.assertTrue((output/f"figures/{figure}.png").exists())
                    svg=(output/f"figures/{figure}.svg").read_text()
                    if language=="en":self.assertNotRegex(svg,r"[\u3400-\u9fff]")
            self.assertEqual(catalogs["zh"]["counts"],catalogs["en"]["counts"])
            self.assertEqual(catalogs["en"]["counts"]["errors"],1)
            for chinese,english in zip(catalogs["zh"]["files"],catalogs["en"]["files"]):
                for key in ["id","path","kind","ll","bic","k","delta_bic","delta_ll","probability_pass"]:
                    self.assertEqual(chinese.get(key),english.get(key))
                self.assertNotRegex(" ".join(english["warnings"]),r"[\u3400-\u9fff]")
                zh_payload=self.read_payload(root/"zh"/chinese["href"])
                en_payload=self.read_payload(root/"en"/english["href"])
                self.assertEqual(zh_payload["record"],en_payload["record"])
                self.assertEqual(zh_payload.get("metadata"),en_payload.get("metadata"))
                if english["kind"]=="model":
                    self.assertEqual(en_payload["record"]["original_note"],result["original_note"])
                    self.assertEqual(en_payload["fit"]["reason"],"Only saved diagnostics were requested")
                    self.assertEqual(en_payload["clusters"][0]["label"],"Class 1")
                    for payload in [zh_payload,en_payload]:
                        cluster=payload["clusters"][0]
                        self.assertEqual(cluster["amplitude_zero_limit"],cluster["values"]["lambda"])
                        self.assertNotIn("amplitude_100",cluster)
                        self.assertEqual(payload["summary"]["clusters"][0]["amplitude_zero_limit"],cluster["amplitude_zero_limit"])
                    for language,summary in [("zh",chinese),("en",english)]:
                        output=root/language
                        for extension,relative in summary["utility_figures"].items():
                            self.assertTrue((output/relative).exists())
                            self.assertEqual(self.read_payload(output/summary["href"])["summary"]["utility_figures"][extension],relative)
                            if extension=="svg" and language=="en":
                                self.assertNotRegex((output/relative).read_text(),r"[\u3400-\u9fff]")
            with (root/"en/parameters.csv").open(encoding="utf-8-sig") as fh:
                parameters=list(csv.DictReader(fh))
            self.assertIn("Fixed",{p["status"] for p in parameters})
            self.assertIn("Linked to alpha_gain",{p["status"] for p in parameters})
            self.assertIn("Near Lower Bound",{p["boundary"] for p in parameters})
            for parameter in parameters:
                self.assertNotRegex(parameter["status"]+parameter["boundary"],r"[\u3400-\u9fff]")
            readme=(root/"en/README.md").read_text()
            self.assertIn("python export_conditional_html.py --language en",readme)
            self.assertNotRegex(readme,r"[\u3400-\u9fff]")
            self.assertEqual((model/"result.pkl").read_bytes(),original)


class UtilityReportTests(unittest.TestCase):
    def test_zero_limit_uses_exact_curvature_order(self):
        for gain,loss,expected in [(1.,2.,0.),(1.,1.,2.25),(2.,1.,float("inf")),
                                  (1.,1.+1e-12,0.),(1.,1.-1e-12,float("inf"))]:
            with self.subTest(gain=gain,loss=loss):
                self.assertEqual(report.amplitude_zero_limit(2.25,gain,loss),expected)

    def test_infinite_zero_limit_is_preserved_in_both_report_languages(self):
        result=ReportLanguageTests.make_result()
        result["thetas"][0][report.CANON[result["spec"].pw]["AG"]]=1.
        result["thetas"][0][report.CANON[result["spec"].pw]["AL"]]=.5
        for language in ["zh","en"]:
            with self.subTest(language=language):
                clusters=report.cluster_parameters(result,1e-4,language)
                self.assertEqual(clusters[0]["amplitude_zero_limit"],float("inf"))
                self.assertEqual(json.loads(report.json_text(clusters))[0]["amplitude_zero_limit"],"Infinity")

    def test_utility_figure_uses_separate_gain_and_loss_curvature(self):
        offsets,utility=report.utility_curve({"alpha_gain":.5,"alpha_loss":2.,"lambda":2.},extent=9)
        self.assertEqual(offsets[0],-9)
        self.assertEqual(offsets[-1],9)
        self.assertEqual(utility[0],-162)
        self.assertEqual(utility[-1],3)
        self.assertEqual(utility[offsets==0].tolist(),[0])
        self.assertTrue(np.all(np.diff(offsets)>0))
        self.assertTrue(np.all(utility[offsets<0]<0))
        self.assertTrue(np.all(utility[offsets>0]>0))


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
        # Audit the archived result against its original participant filter.
        snapshot=report.ROOT/"results_grid_conditional/corrected_source_snapshot/data.py"
        module_spec=importlib.util.spec_from_file_location("conditional_fixture_data",snapshot)
        fixture_data=importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(fixture_data)
        observed, *_=fixture_data.process(export_excel=False)
        data=report.prepare_grid_data(observed,report.f.transform(report.GlobalLottery))
        clusters=report.cluster_parameters(self.result,1e-4)
        fit=report.audit_result(self.result,data,clusters)
        self.assertEqual(fit["status"],"verified")
        self.assertLess(abs(fit["ll_gap"]),1e-5)
        self.assertLess(fit["posterior_gap"],1e-7)
        self.assertEqual(len(fit["states"]),40)
        self.assertEqual([row["n"] for row in fit["stage"]],[18096,1130,1131])
        self.assertTrue(all(c["negative_states"]==0 for c in clusters))


if __name__=="__main__":unittest.main()
