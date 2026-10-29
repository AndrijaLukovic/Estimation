"""Export conditional-grid pickles to a bounded, offline HTML explorer.

Inputs are locally generated result/checkpoint pickles; the exporter never
estimates models or changes any input. Results are recomputed (audited) when
the directory protocol's inputs match the current data and settings.
"""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory

from ..config import LOTTERIES, ROOT
from ..data import observed_ce
from ..lottery_paths import transform
from ..prepare import prepare_grid_data
from ..store import input_differences
from .figures import save_figures
from .html import VERSION, finish_report, install_report_assets
from .i18n import localize
from .serialize import json_text, readable
from .storage import CSVParts, DataBundles, publish_report, validate_combined_reports
from .summary import summarize

PARAMETER_COLUMNS = ["file", "config_id", "kind", "cluster", "pi", "name", "value", "status", "lower", "upper", "boundary"]
DIAGNOSTIC_COLUMNS = ["config_id", "kind", "path", "bic", "delta_bic", "ll", "delta_ll", "k", "n_obs", "n_subjects", "converged",
    "exit_reason", "n_iter", "probability_pass", "near_zero_prelec", "boundary_count", "optimizer_unsuccessful",
    "optimizer_rejected", "ll_gap", "posterior_gap", "normalized_rmse", "ksi_median", "posterior_median", "href"]


def default_output_dir(language):
    name = "grid_em_conditional_explorer" + ("_en" if language == "en" else "")
    return ROOT / "reports" / name


def audit_context(protocol, audit):
    """Prepared current data when the protocol's likelihood inputs match the
    current data and settings; else the reason. Also the differing inputs."""
    if not protocol:
        return dict(data=None, reason="未找到目录估计协议"), []
    try:
        observed = observed_ce()
        changed = input_differences(protocol, observed)
    except Exception as exc:
        return dict(data=None, reason=f"无法准备复算数据：{exc}"), []
    if changed:
        return dict(data=None, reason="当前估计输入与协议不同：" + ", ".join(changed)), changed
    if audit == "none":
        return dict(data=None, reason="未找到目录估计协议"), changed
    try:
        return dict(data=prepare_grid_data(observed, transform(LOTTERIES)), reason=None), changed
    except Exception as exc:
        return dict(data=None, reason=f"无法准备复算数据：{exc}"), changed


def _export_file(path, input_dir, protocol, context, args, language, parameters):
    """(summary, detail payload, raw bytes) of one pickle."""
    relative = str(path.relative_to(input_dir))
    blob = path.read_bytes()
    record = pickle.loads(blob)
    meta_path = (path.parent.parent if path.parent.name == "restarts" else path.parent)/"meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    recognized = isinstance(record, dict) and all(k in record for k in ["spec", "thetas", "layouts", "resp", "ksi", "subjects"])
    if not recognized:
        summary = dict(kind="other", config_id=path.stem, path=relative, spec={}, clusters=[],
            warnings=[localize("非标准结果结构", language)], comparison_group=None, probability_pass=None, audit_status="not_run")
        return summary, dict(summary=summary, record=readable(record), clusters=[], diagnostics={}, fit={}, metadata=meta), blob
    do_audit = args.audit == "all" or (args.audit == "models" and path.name == "result.pkl")
    summary, payload = summarize(record, path, input_dir, meta, protocol, context, args.boundary_tol, do_audit, language)
    for cluster in payload["clusters"]:
        for parameter in cluster["parameters"]:
            parameters.add(dict(file=relative, config_id=summary["config_id"], kind=summary["kind"],
                cluster=cluster["cluster"], pi=cluster["pi"], **parameter))
    return summary, payload, blob


def build_export(args, output):
    language = getattr(args, "language", "zh")
    input_dir = args.input_dir.resolve()
    protocol_path = input_dir/"run_protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    context, changed = audit_context(protocol, args.audit)
    paths = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in [".pkl", ".pickle", ".p"])
    if args.models_only:
        paths = [p for p in paths if p.name == "result.pkl"]
    if not paths:
        raise ValueError(localize("未找到 pickle 文件：", language)+str(input_dir))
    output.mkdir(parents=True, exist_ok=True)
    install_report_assets(output, language)
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    summaries, manifest, errors, details = [], [], [], {}
    bundles = DataBundles(output)
    parameters = CSVParts(output, "parameters", PARAMETER_COLUMNS)
    print(f"HTML EXPORT: {len(paths)} pickle files; language={language}; audit={args.audit}; input={input_dir}", flush=True)
    for i, path in enumerate(paths, 1):
        relative = str(path.relative_to(input_dir))
        identifier = hashlib.sha256(relative.encode()).hexdigest()[:16]
        href = f"detail.html?id={identifier}"
        try:
            summary, payload, blob = _export_file(path, input_dir, protocol, context, args, language, parameters)
            summary.update(id=identifier, href=href)
            if summary["kind"] == "model":
                summary["utility_figures"] = {extension: f"figures/utility/{identifier}.{extension}" for extension in ["svg", "png"]}
            payload.update(generated_at=generated, boundary_tolerance=args.boundary_tol, language=language)
            details[identifier] = bundles.add(identifier, json_text(payload))
            summaries.append(summary)
            manifest.append(dict(path=relative, sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob), href=href, status="ok"))
        except Exception as exc:
            error = f"{type(exc).__name__}: {localize(str(exc), language)}"
            errors.append(dict(path=relative, error=error))
            summary = dict(id=identifier, kind="error", config_id=path.stem, path=relative, href=href, spec={}, clusters=[],
                warnings=[localize("转换失败", language)], comparison_group=None, probability_pass=None, audit_status="error")
            details[identifier] = bundles.add(identifier, json_text(dict(summary=summary, error=error, record=None, language=language)))
            summaries.append(summary)
            manifest.append(dict(path=relative, href=href, status="error", error=error))
        if i % 250 == 0 or i == len(paths):
            print(f"  [{i}/{len(paths)}] exported; errors={len(errors)}", flush=True)
    bundles.close()
    parameters.close()

    models = [s for s in summaries if s["kind"] == "model"]
    best_groups = defaultdict(list)
    for s in models:
        if s.get("bic") is not None:
            best_groups[s["comparison_group"]].append(s["bic"])
    config_ll = defaultdict(list)
    for s in summaries:
        if s.get("ll") is not None:
            config_ll[s["config_id"]].append(s["ll"])
    for s in summaries:
        s["delta_bic"] = s["bic"]-min(best_groups[s["comparison_group"]]) if s.get("bic") is not None and s["comparison_group"] in best_groups else None
        s["delta_ll"] = max(config_ll[s["config_id"]])-s["ll"] if s.get("ll") is not None else None
    save_figures(output, summaries, language)

    progress = []
    for path in sorted(input_dir.glob("test*/restarts/*.progress.json")):
        completed = path.with_name(path.name.replace(".progress.json", ".pkl"))
        if not completed.exists():
            try:
                progress.append(dict(path=str(path.relative_to(input_dir)), **json.loads(path.read_text())))
            except (ValueError, OSError):
                pass
    catalog = dict(version=VERSION, language=language, generated_at=generated, input_dir=str(input_dir), files=summaries,
        counts=dict(total=len(paths), models=len(models), restarts=sum(s["kind"] == "restart" for s in summaries),
            errors=len(errors), tests=dict(Counter(s["spec"]["test"] for s in models)), progress_only=len(progress)),
        protocol=protocol, source_matches=bool(protocol) and not changed, source_differences=changed,
        audit_mode=args.audit, boundary_tolerance=args.boundary_tol, progress_only=progress)
    diagnostics = CSVParts(output, "diagnostics", DIAGNOSTIC_COLUMNS)
    for summary in summaries:
        diagnostics.add(summary)
    diagnostics.close()
    finish_report(output, catalog, dict(generated_at=generated, files=manifest, errors=errors, audit_mode=args.audit),
        details, dict(parameters=parameters.parts, diagnostics=diagnostics.parts), language)
    return catalog


def export(args):
    """Build the report in a temporary directory, validate it, then replace the output."""
    language = getattr(args, "language", "zh")
    input_dir = args.input_dir.resolve()
    output = (args.output_dir or default_output_dir(language)).resolve()
    if output == input_dir or input_dir in output.parents:
        raise ValueError(localize("请将 HTML 报告输出到结果目录之外", language))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".report-build-", dir=output.parent) as temporary:
        staged = Path(temporary)/"report"
        catalog = build_export(args, staged)
        if output.is_relative_to(ROOT/"reports"):
            validate_combined_reports(ROOT/"reports", {output: staged})
        publish_report(staged, output)
    print(json.dumps(dict(report=str(output/"index.html"), **catalog["counts"]), ensure_ascii=False), flush=True)
    return catalog
