"""Standalone report figures; interactive plots use the report's JS assets."""
from collections import defaultdict
from pathlib import Path

from .i18n import localize
from .summary import utility_curve


def _short_label(config_id, language):
    return (config_id.replace("commem_", "").replace("commonalpha", localize("共同", language))
            .replace("sepalpha", localize("分开", language)))


def save_figures(output, summaries, language="zh"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    matplotlib.rcParams.update({"font.family": ["DejaVu Sans", "Droid Sans Fallback"],
        "svg.fonttype": "path", "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})
    directory = output/"figures"
    directory.mkdir(exist_ok=True)

    models = [s for s in summaries if s["kind"] == "model" and not s.get("equivalent_to") and s.get("delta_bic") is not None]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, test in zip(axes, ["test1", "test2", "test3"]):
        rows = sorted([s for s in models if s["spec"]["test"] == test], key=lambda s: s["delta_bic"])[:10]
        labels = [_short_label(s["config_id"].replace(test+"_", ""), language) for s in rows]
        values = [s["delta_bic"] for s in rows]
        ax.barh(labels, values, color="#127e7a")
        ax.invert_yaxis()
        ax.set_title(test.upper()+localize(" · 前 10 个保存模型", language))
        ax.set_xlabel(localize("ΔBIC（同测试、同协议与同样本）", language))
        ax.set_xlim(0, max(values+[1])*1.23)
        for i, value in enumerate(values):
            ax.text(value+max(values+[1])*.015, i, f"{value:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(directory/"model_bic_overview.svg", bbox_inches="tight")
    fig.savefig(directory/"model_bic_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    groups = defaultdict(list)
    for s in summaries:
        if s["kind"] == "restart" and s.get("delta_ll") is not None:
            groups[s["config_id"]].append(s["delta_ll"])
    rows = sorted([(key, max(values), len(values)) for key, values in groups.items()], key=lambda row: row[1], reverse=True)[:16]
    fig, ax = plt.subplots(figsize=(11, 7))
    if rows:
        labels = [_short_label(key, language)+f" · {n} "+localize("起点", language) for key, _, n in rows]
        ax.barh(labels, [v for _, v, _ in rows], color="#d49436")
        ax.invert_yaxis()
    else:
        ax.text(.5, .5, localize("本次没有导出起点 pickle", language), ha="center", va="center", transform=ax.transAxes)
    ax.set_title(localize("同配置起点的最大 LL 差距（前 16 个配置）", language))
    ax.set_xlabel(localize("最高已导出 LL − 最低起点 LL", language))
    fig.tight_layout()
    fig.savefig(directory/"restart_ll_spread.svg", bbox_inches="tight")
    fig.savefig(directory/"restart_ll_spread.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    utilities = [summary for summary in summaries if summary.get("utility_figures")]
    for index, summary in enumerate(utilities, 1):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for cluster in summary["clusters"]:
            offsets, utility = utility_curve(cluster["values"])
            ax.plot(offsets, utility, label=cluster["label"])
        ax.axhline(0, color="#8597a5", linewidth=.8, linestyle="--")
        ax.axvline(0, color="#8597a5", linewidth=.8, linestyle="--")
        ax.set_xlim(-100, 100)
        ax.set_title(summary["config_id"]+"\n"+localize("效用函数 u(z)", language))
        ax.set_xlabel(localize("相对参考点的金额 z=x−R（英镑）", language))
        ax.set_ylabel("u(z)")
        ax.legend()
        fig.tight_layout()
        for extension, relative in summary["utility_figures"].items():
            target = output/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, bbox_inches="tight", **({"dpi": 150} if extension == "png" else {}))
        plt.close(fig)
        if index % 25 == 0 or index == len(utilities):
            print(f"  utility figures [{index}/{len(utilities)}]", flush=True)
