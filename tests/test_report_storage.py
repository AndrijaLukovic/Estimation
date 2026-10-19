"""Byte boundaries, lossless splitting, and migration of complete snapshots."""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import export_conditional_html as report
import organize_reports
import report_storage as storage
import test_conditional_html_export as fixtures


class ReportStorageTests(unittest.TestCase):
    def test_large_unicode_record_spans_bundles_without_changing_values(self):
        value={"raw":"参与者😀\n\"</script>"*1000,"numbers":[1e-30,1.2345678901234567],"nonfinite":"Infinity"}
        with TemporaryDirectory() as directory:
            root=Path(directory);bundles=storage.DataBundles(root,part_bytes=1024)
            descriptor=bundles.add("record",report.json_text(value));bundles.close()
            self.assertGreater(len(descriptor["parts"]),1)
            self.assertTrue(all(p.stat().st_size<=1024 for p in root.rglob("*.js")))
            self.assertEqual(storage.read_bundle_value(root,"record",descriptor),value)
            first=root/descriptor["parts"][0]
            first.write_text(first.read_text().replace("参与者","改变了",1))
            with self.assertRaisesRegex(ValueError,"checksum"):
                storage.read_bundle_value(root,"record",descriptor)

    def test_more_than_ten_thousand_records_do_not_require_ten_thousand_files(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);bundles=storage.DataBundles(root);descriptors={}
            for index in range(10_050):
                descriptors[str(index)]=bundles.add(str(index),report.json_text({"value":index}))
            bundles.close()
            self.assertLess(storage.validate_report_tree(root)["files"],10)
            recovered=dict((key,json.loads(source)["value"]) for key,source in storage.iter_bundle_sources(root,descriptors))
            self.assertEqual(len(recovered),10_050)
            self.assertEqual(recovered["10049"],10_049)

    def test_csv_parts_keep_whole_rows_headers_unicode_and_precision(self):
        rows=[{"id":str(i),"text":"中文,含\"引号\"\n换行😀","number":"1.2345678901234567"} for i in range(50)]
        with TemporaryDirectory() as directory:
            root=Path(directory);writer=storage.CSVParts(root,"parameters",list(rows[0]),part_bytes=250)
            for row in rows:writer.add(row)
            writer.close();recovered=[]
            self.assertGreater(len(writer.parts),1)
            for relative in writer.parts:
                path=root/relative;self.assertLessEqual(path.stat().st_size,250)
                with path.open(encoding="utf-8-sig",newline="") as handle:
                    reader=csv.DictReader(handle);self.assertEqual(reader.fieldnames,list(rows[0]));recovered.extend(reader)
            self.assertEqual(recovered,rows)

    def test_catalog_parts_and_combined_limits(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);rows=[{"value":1.2345678901234567,"label":"类别😀"}]*50
            parts=storage.write_json_parts(root,"catalog",rows,part_bytes=200)
            self.assertEqual([row for part in parts for row in json.loads((root/part).read_text())],rows)
            self.assertTrue(all((root/part).stat().st_size<=200 for part in parts))
            for part in parts:(root/part).unlink()
            old=root/"old";old.mkdir();(old/"obsolete").write_text("old")
            staged=root/"staged";staged.mkdir();(staged/"new").write_text("new")
            (root/"other").write_text("preserved")
            with patch.object(storage,"MAX_REPORT_FILES",2):
                self.assertEqual(storage.validate_combined_reports(root,{old:staged}),2)
                (staged/"extra").write_text("extra")
                with self.assertRaisesRegex(ValueError,"Combined reports"):
                    storage.validate_combined_reports(root,{old:staged})
            with (staged/"too_large").open("wb") as handle:handle.truncate(storage.MAX_FILE_BYTES+1)
            with self.assertRaisesRegex(ValueError,"25 MB"):storage.validate_report_tree(staged)

    def test_report_replacement_restores_previous_directory_on_failure(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);old=root/"old";old.mkdir();(old/"catalog.json").write_text("original")
            staged=root/"staged";staged.mkdir();(staged/"catalog.json").write_text("replacement")
            rename=Path.rename
            def fail_move(path,target):
                if path==staged:raise OSError("simulated publication failure")
                return rename(path,target)
            with patch.object(Path,"rename",fail_move):
                with self.assertRaises(OSError):storage.publish_report(staged,old)
            self.assertEqual((old/"catalog.json").read_text(),"original")
            self.assertEqual((staged/"catalog.json").read_text(),"replacement")

    def test_migration_preserves_audits_raw_records_languages_and_unlisted_details(self):
        with TemporaryDirectory() as directory:
            reports=Path(directory)/"reports";reports.mkdir()
            originals={}
            for language in ["zh","en"]:
                root=reports/("explorer"+("_en" if language=="en" else ""));(root/"details").mkdir(parents=True)
                result=fixtures.ReportLanguageTests.make_result()
                summary,payload=report.summarize(result,Path("/inputs/result.pkl"),Path("/inputs"),{}, {},
                    {"data":None,"reason":"未找到目录估计协议"},1e-4,False,language)
                summary.update(id="model",href="details/model.html",delta_bic=0,delta_ll=0)
                payload.update(generated_at="original snapshot",boundary_tolerance=1e-4,language=language)
                payload["fit"]={"status":"verified","ll_gap":0.,"posterior_gap":1e-14,"states":[{"ce":[1.2345678901234567]}]}
                originals[language]=organize_reports.preserved_hash(payload)
                report.write_html(root/summary["href"],"model",payload,"detail",language)
                catalog=dict(version="conditional_html_v1",language=language,files=[summary],input_dir="/inputs",generated_at="original snapshot",
                    counts=dict(total=1,models=1,restarts=0,errors=0,tests={"test1":1},progress_only=0),protocol={},source_matches=True,
                    source_differences=[],audit_mode="all",boundary_tolerance=1e-4,progress_only=[])
                (root/"catalog.json").write_text(report.json_text(catalog))
                (root/"manifest.json").write_text(report.json_text({"files":[{"path":"result.pkl","href":summary["href"]}],"errors":[]}))
                for stem,columns in [("parameters",report.PARAMETER_COLUMNS),("diagnostics",report.DIAGNOSTIC_COLUMNS)]:
                    writer=storage.CSVParts(root,stem,columns)
                    if stem=="diagnostics":writer.add(summary)
                    writer.close()
                if language=="zh":
                    orphan={**payload,"summary":{**summary,"id":"orphan","href":"details/orphan.html","kind":"restart"},"record":{"unlisted":"完整保留"}}
                    report.write_html(root/"details/orphan.html","orphan",orphan,"detail",language)
            with redirect_stdout(io.StringIO()):organize_reports.organize(reports)
            self.assertLess(storage.validate_report_tree(reports)["files"],100)
            for language in ["zh","en"]:
                root=reports/("explorer"+("_en" if language=="en" else ""))
                metadata=json.loads((root/"catalog.json").read_text());self.assertEqual(metadata["language"],language)
                details=metadata["storage"]["details"]
                payload=storage.read_bundle_value(root,"model",details["model"])
                self.assertEqual(organize_reports.preserved_hash(payload),originals[language])
                self.assertFalse((root/"details").exists())
                if language=="zh":
                    self.assertEqual(metadata["counts"]["total"],2)
                    self.assertEqual(storage.read_bundle_value(root,"orphan",details["orphan"])["record"],{"unlisted":"完整保留"})


if __name__=="__main__":unittest.main()
