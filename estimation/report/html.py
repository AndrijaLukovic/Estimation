"""Report pages, assets, catalog and README of an offline report directory."""
from html import escape
import shutil

from .i18n import TEMPLATES, localize
from .serialize import embed_json, json_text, readable
from .storage import (DataBundles, MAX_FILE_BYTES, MAX_REPORT_FILES, iter_bundle_sources,
                      validate_report_tree, write_json_parts)

VERSION = "conditional_html_v2"
EXPORT_COMMAND = "python scripts/export_report.py"


def write_html(path, title, payload, page, language="zh"):
    body = localize((TEMPLATES / f"{page}.html").read_text(encoding="utf-8"), language)
    prefix = "assets/"
    locale = "en" if language == "en" else "zh-CN"
    html = (f"<!doctype html><html lang=\"{locale}\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{escape(localize(title, language))}</title><link rel=\"stylesheet\" href=\"{prefix}style.css\">"
            f"</head><body data-page=\"{page}\">{body}"
            f"<script id=\"report-data\" type=\"application/json\">{embed_json(payload)}</script>"
            f"<script src=\"{prefix}loader.js\"></script><script src=\"{prefix}app.js\"></script></body></html>")
    temporary = path.with_suffix(".tmp.html")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(path)


def install_report_assets(output, language):
    (output/"assets").mkdir(parents=True, exist_ok=True)
    for name in ["style.css", "loader.js"]:
        shutil.copyfile(TEMPLATES/name, output/"assets"/name)
    (output/"assets"/"app.js").write_text(localize((TEMPLATES/"app.js").read_text(encoding="utf-8"), language), encoding="utf-8")


README = (
    "# Grid EM Conditional 独立交互报告\n\n打开 index.html；保留 assets、data 和 figures 目录即可离线使用。所有记录共用 detail.html，网址中的 id 选择对应 pickle。\n\n"
    "全部原始字段、个人参数、后验、迭代日志与复算结果完整保留，数值不取整。数据按字节分块；CSV 每个分块重复表头，catalog.json 列出全部目录 JSON 分块。单文件不超过 25,000,000 字节，整个 reports 目录不超过 10,000 个文件。\n\n"
    "重新运行 {command} 更新结果快照；运行中的估计不会被停止或修改。默认复算全部 pickle，--audit models 只复算模型，--audit none 只读取保存诊断。--models-only 仅导出模型 pickle。\n\n"
    "ΔBIC 只在相同测试、协议、参与者集合及观测数内比较；等价标签可隐藏。ΔLL 是同配置已导出起点距最高 LL 的差。单类后验为 1 由模型约束决定。模型边界、局部解与无标准误仍限制结构解释。\n\n"
    "选择模型或勾选比较即可查看各类别的效用函数。详情页可调整绘图范围；模型结果另提供默认 ±£100 范围的 SVG/PNG。类别总览显示 A(d) 在 d→0⁺ 时的极限：α⁻>α⁺ 为 0，α⁻=α⁺ 为 λ，α⁻<α⁺ 为 +∞。\n\n"
    "超大浏览器下载会拆成 .part 文件及 .parts.json 清单；按清单顺序以二进制连接各 part，即可无损还原原文件。\n")


def finish_report(output, catalog, manifest, details, downloads, language):
    """Write small entry pages and bounded downloadable catalog/CSV parts."""
    catalog.update(version=VERSION, language=language)
    for summary in catalog["files"]:
        details[summary["id"]]["summary_extra"] = {key: summary.get(key) for key in ["delta_bic", "delta_ll"]}
    bundles = DataBundles(output, stem="catalog")
    descriptor = bundles.add("catalog", json_text(catalog["files"]))
    bundles.close()
    catalog["storage"] = dict(format="json-script-chunks-v1", catalog=descriptor, details=details,
        max_file_bytes=MAX_FILE_BYTES, max_report_files=MAX_REPORT_FILES)
    downloads["catalog"] = write_json_parts(output, "catalog", readable(catalog["files"]))
    catalog["downloads"] = downloads
    metadata = {key: value for key, value in catalog.items() if key != "files"}
    write_html(output/"index.html", "Grid EM Conditional · 交互结果报告", metadata, "dashboard", language)
    write_html(output/"detail.html", "Grid EM Conditional · 文件详情", metadata, "detail", language)
    (output/"catalog.json").write_text(json_text(metadata), encoding="utf-8")
    manifest.update(version=VERSION, language=language, storage_format="json-script-chunks-v1")
    (output/"manifest.json").write_text(json_text(manifest), encoding="utf-8")
    command = EXPORT_COMMAND + (" --language en" if language == "en" else "")
    (output/"README.md").write_text(localize(README.format(command=command), language), encoding="utf-8")
    # Verify every serialized record before accepting the report.
    for _ in iter_bundle_sources(output, details):
        pass
    for _ in iter_bundle_sources(output, {"catalog": descriptor}):
        pass
    return validate_report_tree(output)
