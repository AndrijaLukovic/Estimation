"""Compact existing reports losslessly, preserving saved audits and languages.

Usage:
    python organize_reports.py
    python organize_reports.py --check
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

import export_conditional_html as report
from report_storage import (CSVParts, DataBundles, iter_bundle_sources, publish_report,
                            validate_combined_reports, validate_report_tree)

PAYLOAD = re.compile(r'<script id="report-data" type="application/json">(.*?)</script>', re.S)
PRESERVED = ("record", "metadata", "diagnostics", "fit", "generated_at", "boundary_tolerance")


def preserved_hash(payload):
    data={key:payload[key] for key in PRESERVED if key in payload}
    # Saved JSON already contains only JSON types. Avoid walking millions of
    # scalar entries through the pickle/NumPy conversion a second time.
    return hashlib.sha256(saved_json(data).encode("utf-8")).hexdigest()


def saved_json(value):
    return json.dumps(value,ensure_ascii=False,separators=(",",":"),allow_nan=False)


def upgrade_clusters(clusters):
    for cluster in clusters:
        values=cluster.get("values",{})
        if all(key in values for key in ["lambda","alpha_gain","alpha_loss"]):
            cluster["amplitude_zero_limit"]=report.readable(report.amplitude_zero_limit(
                values["lambda"],values["alpha_gain"],values["alpha_loss"]))
        cluster.pop("amplitude_100",None)


def migrate_report(source, staged):
    catalog=json.loads((source/"catalog.json").read_text(encoding="utf-8"))
    language=catalog.get("language","en" if source.name.endswith("_en") else "zh")
    catalog["language"]=language
    manifest=json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    staged.mkdir(parents=True)
    report.install_report_assets(staged,language)
    if (source/"figures").exists():shutil.copytree(source/"figures",staged/"figures")
    known={source/entry["href"] for entry in catalog["files"]}
    archived=[]
    for path in sorted((source/"details").glob("*.html")):
        if path not in known:
            match=PAYLOAD.search(path.read_text(encoding="utf-8"))
            if match is None:raise ValueError(f"Missing report payload: {path}")
            payload=json.loads(match.group(1));summary=payload["summary"]
            summary.setdefault("id",path.stem);summary["href"]=str(path.relative_to(source))
            summary["archived_snapshot"]=True
            catalog["files"].append(summary);archived.append(payload)
            manifest["files"].append(dict(path=summary["path"],href=summary["href"],status="archived",
                report_page_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    bundles=DataBundles(staged)
    details={};hashes={};hrefs={}
    for index,summary in enumerate(catalog["files"],1):
        path=source/summary["href"]
        match=PAYLOAD.search(path.read_text(encoding="utf-8"))
        if match is None:raise ValueError(f"Missing report payload: {path}")
        payload=json.loads(match.group(1));identifier=summary["id"]
        hashes[identifier]=preserved_hash(payload)
        previous_href=summary["href"]
        summary["href"]=f"detail.html?id={identifier}";hrefs[previous_href]=summary["href"]
        upgrade_clusters(summary.get("clusters",[]));upgrade_clusters(payload.get("clusters",[]))
        if summary["kind"]=="model":
            summary["utility_figures"]={ext:f"figures/utility/{identifier}.{ext}" for ext in ["svg","png"]}
        payload.update(summary=summary,language=language)
        details[identifier]=bundles.add(identifier,saved_json(payload))
        if index%1000==0 or index==len(catalog["files"]):
            print(f"  {source.name}: packed {index}/{len(catalog['files'])} records",flush=True)
    bundles.close()
    for entry in manifest["files"]:entry["href"]=hrefs.get(entry["href"],entry["href"])
    downloads={}
    for stem in ["parameters","diagnostics"]:
        with (source/f"{stem}.csv").open(encoding="utf-8-sig",newline="") as handle:
            reader=csv.DictReader(handle);writer=CSVParts(staged,stem,reader.fieldnames)
            for row in reader:
                if stem=="diagnostics":row["href"]=hrefs.get(row.get("href"),row.get("href"))
                writer.add(row)
            for payload in archived:
                summary=next(item for item in catalog["files"] if item["id"]==payload["summary"]["id"])
                if stem=="diagnostics":writer.add(summary)
                else:
                    for cluster in payload.get("clusters",[]):
                        for parameter in cluster.get("parameters",[]):
                            writer.add(dict(file=summary["path"],config_id=summary["config_id"],kind=summary["kind"],
                                cluster=cluster["cluster"],pi=cluster["pi"],**parameter))
            writer.close();downloads[stem]=writer.parts
    if archived:
        counts=catalog["counts"];counts["total"]=len(catalog["files"])
        for kind,plural in [("model","models"),("restart","restarts"),("error","errors")]:
            counts[plural]=sum(item["kind"]==kind for item in catalog["files"])
        counts["tests"]=dict(Counter(item["spec"]["test"] for item in catalog["files"] if item["kind"]=="model"))
        catalog["archived_snapshots"]=len(archived)
    report.save_figures(staged,catalog["files"],language)
    report.finish_report(staged,catalog,manifest,details,downloads,language)
    for key,serialized in iter_bundle_sources(staged,details):
        if preserved_hash(json.loads(serialized))!=hashes[key]:
            raise ValueError(f"Original fields or audit results changed: {key}")
    print(f"  {source.name}: verified {len(hashes)} complete records",flush=True)
    return validate_report_tree(staged)


def organize(reports_dir):
    reports_dir=Path(reports_dir).resolve()
    if not reports_dir.is_dir():raise ValueError(f"Reports directory does not exist: {reports_dir}")
    with TemporaryDirectory(prefix=".reports-organize-",dir=reports_dir.parent) as temporary:
        replacements={}
        for path in sorted(reports_dir.glob("*/catalog.json")):
            metadata=json.loads(path.read_text(encoding="utf-8"))
            if not str(metadata.get("version","")).startswith("conditional_html_") or metadata.get("storage"):
                continue
            source=path.parent;staged=Path(temporary)/source.name
            migrate_report(source,staged);replacements[source]=staged
        validate_combined_reports(reports_dir,replacements)
        for source,staged in replacements.items():publish_report(staged,source)
    return validate_report_tree(reports_dir)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir",type=Path,default=report.ROOT/"reports")
    parser.add_argument("--check",action="store_true",help="verify the combined count and byte limits without changing files")
    args=parser.parse_args()
    result=validate_report_tree(args.reports_dir) if args.check else organize(args.reports_dir)
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
