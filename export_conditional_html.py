"""Export every conditional-grid pickle to an independent, offline HTML explorer.

Usage:
    python export_conditional_html.py
    python export_conditional_html.py --audit none     # saved diagnostics only
    python export_conditional_html.py --models-only    # omit restart pickles

Inputs are this project's locally generated result/checkpoint pickle files.
The exporter never estimates models or changes any input result.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime
import hashlib
from html import escape
import json
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

import functions as f
from GlobalSettings import GlobalLottery, GlobalKsiBounds, GlobalRSQMode
from Mixture import _s_to_a
from grid_em import prepare_grid_data, count_free_params
from main import get_observed_ce
from model_spec import CANON, bounds_free

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "report_templates" / "grid_conditional"
VERSION = "conditional_html_v1"
RP = ["SQ", "PA", "LE", "FE"]
NAMES = dict(R="r", AG="alpha_gain", AL="alpha_loss", LAM="lambda", GAM="gamma",
             BETA="beta", PALPHA="palpha", S1="s1", S2="s2", S3="s3", DELTA="delta", PSI="psi")


def readable(value):
    """Keep all fields/array entries; represent nonfinite raw values explicitly."""
    if is_dataclass(value): return readable(asdict(value))
    if isinstance(value, np.ndarray): return readable(value.tolist())
    if isinstance(value, np.generic): return readable(value.item())
    if isinstance(value, dict): return {str(k): readable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [readable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if value is None or isinstance(value, (bool, int, float, str)): return value
    if hasattr(value, "__dict__"): return {"__class__": type(value).__name__, **readable(vars(value))}
    return repr(value)


def number(value):
    try:
        val = float(value)
        return val if np.isfinite(val) else None
    except (TypeError, ValueError): return None


def json_text(value):
    return json.dumps(readable(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def embed_json(value):
    # JSON remains valid while strings cannot close the containing script tag.
    return json_text(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def write_html(path, title, payload, page):
    body = (ASSETS / f"{page}.html").read_text(encoding="utf-8")
    prefix = "../assets/" if page == "detail" else "assets/"
    html = ("<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{escape(title)}</title><link rel=\"stylesheet\" href=\"{prefix}style.css\">"
            f"</head><body data-page=\"{page}\">{body}"
            f"<script id=\"report-data\" type=\"application/json\">{embed_json(payload)}</script>"
            f"<script src=\"{prefix}app.js\"></script></body></html>")
    temporary = path.with_suffix(".tmp.html")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(path)


def probability_shape(method, parameter):
    if parameter is None or parameter <= 0: return False, None
    if method == "prelec": return True, None
    if method != "tk": return None, None
    g = parameter
    if g == 1: return True, 1.0
    def numerator(p):
        return (g-1)*p**g + (1-p)**(g-1)*(g+(1-g)*p)
    minimum = min(g, minimize_scalar(numerator, bounds=(1e-14, 1-1e-14), method="bounded").fun)
    return bool(minimum >= -1e-12), float(minimum)


def reference_points(theta, method, delta_le, data):
    ix = CANON[method]
    a1, a2, a3 = _s_to_a(*(theta[ix[k]] for k in ["S1", "S2", "S3"]))
    a4 = 1-a1-a2-a3
    el, fe, c1, stage, z1, z2, zt = data.reference_inputs
    delta = theta[ix["DELTA"]]
    dle = delta if delta_le is None else delta_le
    pa = np.where(stage == 1, z1/(delta+1), np.where(stage == 2,
         (delta*z1+z2)/(delta**2+delta+1), 0))
    le = np.where(stage == 1, el, np.where(stage == 2, (dle*el+c1)/(dle+1), 0))
    sq = zt if GlobalRSQMode == "zt" else np.zeros_like(zt)
    return a1*sq + a2*pa + a3*le + a4*fe


def cluster_parameters(result, tolerance):
    method = result["spec"].pw
    ix = CANON[method]
    resp = np.asarray(result["resp"], dtype=float)
    clusters = []
    for j, (theta, dle, layout, fv) in enumerate(zip(result["thetas"], result["delta_les"],
                                                   result["layouts"], result["free_vecs"])):
        values = {NAMES[k]: float(theta[pos]) for k, pos in ix.items() if k != "LEN"}
        a1, a2, a3 = _s_to_a(*(values[k] for k in ["s1", "s2", "s3"]))
        weights = [float(a1), float(a2), float(a3), float(1-a1-a2-a3)]
        bounds = bounds_free(layout)
        free_bounds = {pos: bounds[i] for i, pos in enumerate(layout.free_positions)}
        params = []
        for key, pos in ix.items():
            if key == "LEN": continue
            status = "自由估计" if pos in layout.free_positions else "固定"
            if key == "AL" and layout.link_AL_to_AG: status = "与 alpha_gain 绑定"
            if key == "DELTA" and weights[1]+weights[2] <= 1e-12: status += "；当前参考点未使用"
            bound = free_bounds.get(pos)
            value = float(theta[pos])
            side = None
            if bound:
                if abs(value-bound[0]) <= tolerance: side = "下界附近"
                elif abs(value-bound[1]) <= tolerance: side = "上界附近"
            params.append(dict(name=NAMES[key], value=value, status=status,
                lower=bound[0] if bound else None, upper=bound[1] if bound else None, boundary=side))
        effective_le = values["delta"] if dle is None else float(dle)
        le_bound = bounds[-1] if layout.has_delta_le else None
        le_side = None
        if le_bound:
            if abs(effective_le-le_bound[0]) <= tolerance: le_side = "下界附近"
            elif abs(effective_le-le_bound[1]) <= tolerance: le_side = "上界附近"
        params.append(dict(name="delta_le", value=effective_le,
            status=("自由估计" if layout.has_delta_le else "共享 delta" if dle is None else "固定")+
                   ("；当前参考点未使用" if weights[2] <= 1e-12 else ""),
            lower=le_bound[0] if le_bound else None, upper=le_bound[1] if le_bound else None, boundary=le_side))
        parameter = values.get("gamma", values.get("palpha"))
        monotone, minimum = probability_shape(method, parameter)
        identity = result["rp_identities"][j]
        clusters.append(dict(cluster=j+1, label=identity or f"第 {j+1} 类", pi=float(result["pis"][j]),
            soft_n=float(resp[:, j].sum()), map_n=int(np.sum(resp.argmax(axis=1) == j)),
            parameters=params, values=values, weights=weights, effective_delta_le=effective_le,
            free_vec=readable(fv), method=method, probability_parameter=parameter,
            monotone=monotone, derivative_numerator_min=minimum,
            near_zero_prelec=method == "prelec" and parameter-1e-6 <= tolerance,
            boundary_count=sum(p["boundary"] is not None for p in params),
            amplitude_100=values["lambda"] * 100**(values["alpha_loss"]-values["alpha_gain"])))
    return clusters


def saved_diagnostics(result, tolerance):
    ksi = np.asarray(result["ksi"], dtype=float)
    resp = np.asarray(result["resp"], dtype=float)
    n, c = resp.shape
    ll_values = [number(row.get("log_likelihood")) for row in result.get("iter_log", [])]
    ll_values = [v for v in ll_values if v is not None]
    final_ll = number(result.get("log_likelihood"))
    changes = np.diff(ll_values + ([final_ll] if final_ll is not None else []))
    optimizer = result.get("optimizer_log", [])
    maxpost = resp.max(axis=1)
    entropy = -np.sum(resp*np.log(np.maximum(resp, 1e-300)), axis=1)
    return dict(ksi_quantiles=np.quantile(ksi, [0,.05,.25,.5,.75,.95,1]).tolist(),
        ksi_at_lower=int(np.sum(np.abs(ksi-GlobalKsiBounds[0]) <= tolerance)),
        ksi_at_upper=int(np.sum(np.abs(ksi-GlobalKsiBounds[1]) <= tolerance)),
        ksi_bounds=list(GlobalKsiBounds), posterior_row_sum_gap=float(np.max(np.abs(resp.sum(axis=1)-1))),
        posterior_finite=bool(np.isfinite(resp).all()),
        posterior_in_unit_interval=bool(np.all((resp>=0) & (resp<=1))),
        posterior_median=float(np.median(maxpost)) if c>1 else None,
        posterior_below_08=float(np.mean(maxpost<.8)) if c>1 else None,
        normalized_entropy_mean=float(entropy.mean()/np.log(c)) if c>1 else None,
        pi_posterior_gap=float(np.max(np.abs(np.asarray(result["pis"])-resp.mean(axis=0)))),
        ll_decreases=int(np.sum(changes < -1e-6)),
        last_logged_ll_change=float(np.diff(ll_values)[-1]) if len(ll_values)>1 else None,
        final_recomputation_change=number(result.get("final_ll_change")),
        optimizer_calls=len(optimizer), optimizer_unsuccessful=sum(not r.get("success", False) for r in optimizer),
        optimizer_rejected=sum(not r.get("accepted", False) for r in optimizer),
        optimizer_q_increases=sum(number(r.get("q_after")) is not None and number(r.get("q_before")) is not None
            and r["q_after"]>r["q_before"]+1e-8 for r in optimizer),
        optimizer_messages=dict(Counter(str(r.get("message", "")) for r in optimizer)),
        n_subjects=n)


def audit_result(result, data, clusters):
    """Use the shared scalar valuation, including actual decision weights."""
    if result["subjects"] != data.subjects: raise ValueError("参与者集合与当前数据不一致")
    method = result["spec"].pw
    state_mu = np.empty((len(data.state_y), len(clusters)))
    references = np.empty_like(state_mu)
    for j, (theta, dle, cluster) in enumerate(zip(result["thetas"], result["delta_les"], clusters)):
        rr = reference_points(theta, method, dle, data)
        references[:, j] = rr
        v = cluster["values"]
        minimum = np.inf; negative_states = 0
        for i, (_, _, current) in enumerate(data.state_y.attrs["_ce_state_groups"]):
            ev = f.evaluation(r=v["r"], R=float(rr[i]), alpha=v["alpha_gain"],
                alpha_loss=v["alpha_loss"], lamb=v["lambda"], gamma=v.get("gamma",1),
                beta=v.get("beta",1), palpha=v.get("palpha",1), psi=v["psi"], method=method,
                lotteries={"current":current})["current"]
            state_mu[i,j] = f.u_inv(ev["V"],float(rr[i]),v["alpha_gain"],v["lambda"],alpha_loss=v["alpha_loss"])
            outcomes=defaultdict(float)
            for utility,p in zip(ev["present_values_of_streams"],ev["probabilities_of_streams"]):outcomes[utility]+=p
            weights,_=f.dw(outcomes,gamma=v.get("gamma",1),beta=v.get("beta",1),palpha=v.get("palpha",1),method=method)
            smallest=min(weights.values());minimum=min(minimum,smallest)
            negative_states+=int(smallest < -1e-10)
        cluster.update(min_decision_weight=float(minimum), negative_states=negative_states)
    if not np.isfinite(state_mu).all(): raise ValueError("理论 CE 出现非有限值")
    row_mu=state_mu[data.state_index]
    sigma=np.asarray(result["ksi"])[data.subject_index]*data.spread
    pdf=-.5*((data.observed[:,None]-row_mu)/sigma[:,None])**2-np.log(sigma[:,None])-.5*np.log(2*np.pi)
    log_l=np.column_stack([np.bincount(data.subject_index,weights=pdf[:,j],minlength=len(data.subjects)) for j in range(len(clusters))])
    joint=log_l+np.log(np.maximum(result["pis"],1e-300))[None,:]
    sums=logsumexp(joint,axis=1);posterior=np.exp(joint-sums[:,None])
    residual=data.observed-np.sum(row_mu*posterior[data.subject_index],axis=1)
    stages=data.reference_inputs[3][data.state_index]
    stage_stats=[]
    for stage in [0,1,2]:
        mask=stages==stage
        stage_stats.append(dict(stage=stage,n=int(mask.sum()),bias=float(np.mean(residual[mask])),
            rmse=float(np.sqrt(np.mean(residual[mask]**2))),
            normalized_rmse=float(np.sqrt(np.mean((residual[mask]/data.spread[mask])**2)))))
    states=[]
    for i,(key,_,_) in enumerate(data.state_y.attrs["_ce_state_groups"]):
        mask=data.state_index==i
        states.append(dict(lottery=key[0],stage=int(key[1]),z1=float(key[2]),z2=float(key[3]),n=int(mask.sum()),
            observed_mean=float(data.observed[mask].mean()),conditional_expectation=float(data.reference_inputs[1][i]),
            reference=references[i].tolist(),ce=state_mu[i].tolist()))
    return dict(status="verified",ll_recomputed=float(sums.sum()),ll_gap=float(sums.sum()-result["log_likelihood"]),
        posterior_gap=float(np.max(np.abs(posterior-result["resp"]))),
        normalized_rmse=float(np.sqrt(np.mean((residual/data.spread)**2))),
        rmse=float(np.sqrt(np.mean(residual**2))),stage=stage_stats,states=states)


def summarize(result, path, input_dir, meta, protocol, context, tolerance, do_audit):
    spec=asdict(result["spec"])
    clusters=cluster_parameters(result,tolerance)
    saved=saved_diagnostics(result,tolerance)
    version=result.get("valuation_version")
    fingerprint=result.get("run_fingerprint",meta.get("run_fingerprint",protocol.get("fingerprint")))
    fit=dict(status="not_run",reason="选择仅使用已保存诊断")
    if do_audit:
        if not context["data"]:fit=dict(status="not_run",reason=context["reason"])
        elif version!="conditional_remaining_v1":fit=dict(status="not_run",reason="pickle 的估值版本不是当前条件剩余彩票版本")
        elif fingerprint!=protocol.get("fingerprint"):fit=dict(status="not_run",reason="pickle 的协议指纹与目录协议不同")
        else:
            try:fit=audit_result(result,context["data"],clusters)
            except Exception as exc:fit=dict(status="error",reason=f"{type(exc).__name__}: {exc}")
    n_obs=int(result.get("n_obs_effective",result.get("n_obs",0)))
    k=count_free_params(result)
    ll=float(result["log_likelihood"])
    bic=k*np.log(n_obs)-2*ll if n_obs>0 else None
    kind="model" if path.name=="result.pkl" else "restart"
    warnings=[]
    if not result.get("converged"):warnings.append("未达到 LL 容差")
    if any(c["monotone"] is False for c in clusters):warnings.append("概率函数非单调")
    if any(c.get("negative_states",0)>0 for c in clusters):warnings.append("实际彩票有负决策权重")
    if any(c["near_zero_prelec"] for c in clusters):warnings.append("近零 Prelec 参数")
    if any(c["boundary_count"] for c in clusters):warnings.append("自由参数接近边界")
    if saved["optimizer_unsuccessful"]:warnings.append("存在优化器非成功退出")
    if saved["optimizer_rejected"]:warnings.append("存在被拒绝的 M 步")
    if saved["ll_decreases"]:warnings.append("记录中 LL 曾下降")
    if any(c["pi"]<.01 for c in clusters):warnings.append("存在小于 1% 的类别")
    if fit["status"]!="verified":warnings.append("未完成当前数据复算")
    elif abs(fit["ll_gap"])>1e-5:warnings.append("保存 LL 与复算不一致")
    probability_pass=all(c["monotone"] for c in clusters)
    if fit["status"]=="verified":probability_pass=probability_pass and all(c["negative_states"]==0 for c in clusters)
    else:probability_pass=None if probability_pass else False
    subject_hash=hashlib.sha256(json_text(result["subjects"]).encode()).hexdigest()[:16]
    group=f"{fingerprint}|{version}|{n_obs}|{subject_hash}|{spec['test']}"
    brief=[{key:c.get(key) for key in ["cluster","label","pi","soft_n","map_n","weights","values",
        "method","probability_parameter","monotone","near_zero_prelec","boundary_count","amplitude_100",
        "min_decision_weight","negative_states"]} for c in clusters]
    summary=dict(config_id=result["spec"].config_id(),kind=kind,restart_id=result.get("restart_id"),
        path=str(path.relative_to(input_dir)),spec=spec,ll=ll,bic=number(bic),aic=2*k-2*ll,k=k,
        bic_saved=number(result.get("bic")),bic_gap=number(result.get("bic",bic)-bic) if bic is not None else None,
        n_obs=n_obs,n_subjects=len(result["subjects"]),converged=bool(result.get("converged")),
        exit_reason=result.get("exit_reason"),n_iter=result.get("n_iter"),elapsed=number(result.get("elapsed")),
        model_total_elapsed=number(meta.get("elapsed_seconds")) if kind=="model" else None,
        equivalent_to=result.get("equivalent_to",meta.get("equivalent_to")),
        probability_pass=probability_pass,near_zero_prelec=any(c["near_zero_prelec"] for c in clusters),
        boundary_count=sum(c["boundary_count"] for c in clusters),optimizer_unsuccessful=saved["optimizer_unsuccessful"],
        optimizer_rejected=saved["optimizer_rejected"],ll_decreases=saved["ll_decreases"],
        ksi_median=saved["ksi_quantiles"][3],posterior_median=saved["posterior_median"],
        audit_status=fit["status"],ll_gap=fit.get("ll_gap"),posterior_gap=fit.get("posterior_gap"),
        normalized_rmse=fit.get("normalized_rmse"),warnings=warnings,clusters=brief,
        comparison_group=group,fingerprint=fingerprint,valuation_version=version)
    return summary,dict(summary=summary,clusters=clusters,diagnostics=saved,fit=fit,
        metadata=meta,record=readable(result))


def save_figures(output, summaries):
    """Standalone scientific figures; interactive plots use the local JS assets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    if font.exists():font_manager.fontManager.addfont(str(font))
    matplotlib.rcParams.update({"font.family":["DejaVu Sans","Droid Sans Fallback"],
        "svg.fonttype":"path","axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False})
    directory=output/"figures";directory.mkdir(exist_ok=True)
    models=[s for s in summaries if s["kind"]=="model" and not s.get("equivalent_to") and s.get("delta_bic") is not None]
    fig,axes=plt.subplots(1,3,figsize=(16,5.5))
    for ax,test in zip(axes,["test1","test2","test3"]):
        rows=sorted([s for s in models if s["spec"]["test"]==test],key=lambda s:s["delta_bic"])[:10]
        labels=[s["config_id"].replace(test+"_","").replace("commem_","").replace("commonalpha","共同").replace("sepalpha","分开") for s in rows]
        values=[s["delta_bic"] for s in rows]
        ax.barh(labels,values,color="#127e7a");ax.invert_yaxis();ax.set_title(test.upper()+" · 前 10 个保存模型")
        ax.set_xlabel("ΔBIC（同测试、同协议与同样本）")
        ax.set_xlim(0,max(values+[1])*1.23)
        for i,value in enumerate(values):ax.text(value+max(values+[1])*.015,i,f"{value:.2f}",va="center",fontsize=8)
    fig.tight_layout();fig.savefig(directory/"model_bic_overview.svg",bbox_inches="tight");fig.savefig(directory/"model_bic_overview.png",dpi=150,bbox_inches="tight");plt.close(fig)
    groups=defaultdict(list)
    for s in summaries:
        if s["kind"]=="restart" and s.get("delta_ll") is not None:groups[s["config_id"]].append(s["delta_ll"])
    rows=sorted([(key,max(values),len(values)) for key,values in groups.items()],key=lambda row:row[1],reverse=True)[:16]
    fig,ax=plt.subplots(figsize=(11,7))
    if rows:
        labels=[key.replace("commem_","").replace("commonalpha","共同").replace("sepalpha","分开")+f" · {n} 起点" for key,_,n in rows]
        ax.barh(labels,[v for _,v,_ in rows],color="#d49436");ax.invert_yaxis()
    else:ax.text(.5,.5,"本次没有导出起点 pickle",ha="center",va="center",transform=ax.transAxes)
    ax.set_title("同配置起点的最大 LL 差距（前 16 个配置）");ax.set_xlabel("最高已导出 LL − 最低起点 LL")
    fig.tight_layout();fig.savefig(directory/"restart_ll_spread.svg",bbox_inches="tight");fig.savefig(directory/"restart_ll_spread.png",dpi=150,bbox_inches="tight");plt.close(fig)


def export(args):
    input_dir=args.input_dir.resolve();output=args.output_dir.resolve()
    if output==input_dir or input_dir in output.parents:
        raise ValueError("请将 HTML 报告输出到结果目录之外")
    protocol_path=input_dir/"run_protocol.json"
    protocol=json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    changed=[name for name,digest in protocol.get("source_hashes",{}).items()
        if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    context=dict(data=None,reason="未找到目录估计协议")
    if changed:context["reason"]="当前估计源码／数据与协议不同："+", ".join(changed)
    elif protocol and args.audit!="none":
        previous_cwd=Path.cwd()
        try:
            os.chdir(ROOT)
            context=dict(data=prepare_grid_data(get_observed_ce(export_excel=False),f.transform(GlobalLottery)),reason=None)
        except Exception as exc:context["reason"]=f"无法准备复算数据：{exc}"
        finally:os.chdir(previous_cwd)
    paths=sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in [".pkl",".pickle",".p"])
    if args.models_only:paths=[p for p in paths if p.name=="result.pkl"]
    if not paths:raise ValueError(f"未找到 pickle 文件：{input_dir}")
    output.mkdir(parents=True,exist_ok=True);(output/"details").mkdir(exist_ok=True);(output/"assets").mkdir(exist_ok=True)
    for name in ["style.css","app.js"]:shutil.copyfile(ASSETS/name,output/"assets"/name)
    generated=datetime.now().astimezone().isoformat(timespec="seconds")
    summaries=[];manifest=[];errors=[];parameters=[]
    print(f"HTML EXPORT: {len(paths)} pickle files; audit={args.audit}; input={input_dir}",flush=True)
    for i,path in enumerate(paths,1):
        relative=str(path.relative_to(input_dir));identifier=hashlib.sha256(relative.encode()).hexdigest()[:16]
        href=f"details/{identifier}.html"
        try:
            blob=path.read_bytes();record=pickle.loads(blob)
            meta_path=(path.parent.parent if path.parent.name=="restarts" else path.parent)/"meta.json"
            meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
            recognized=isinstance(record,dict) and all(k in record for k in ["spec","thetas","layouts","resp","ksi","subjects"])
            if recognized:
                do_audit=args.audit=="all" or (args.audit=="models" and path.name=="result.pkl")
                summary,payload=summarize(record,path,input_dir,meta,protocol,context,args.boundary_tol,do_audit)
                for cluster in payload["clusters"]:
                    for parameter in cluster["parameters"]:
                        parameters.append(dict(file=relative,config_id=summary["config_id"],kind=summary["kind"],
                            cluster=cluster["cluster"],pi=cluster["pi"],**parameter))
            else:
                summary=dict(kind="other",config_id=path.stem,path=relative,spec={},clusters=[],warnings=["非标准结果结构"],
                    comparison_group=None,probability_pass=None,audit_status="not_run")
                payload=dict(summary=summary,record=readable(record),clusters=[],diagnostics={},fit={},metadata=meta)
            summary.update(id=identifier,href=href)
            payload.update(generated_at=generated,boundary_tolerance=args.boundary_tol)
            write_html(output/href,f"{summary['config_id']} · {path.stem}",payload,"detail")
            summaries.append(summary)
            manifest.append(dict(path=relative,sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),href=href,status="ok"))
        except Exception as exc:
            error=f"{type(exc).__name__}: {exc}"
            errors.append(dict(path=relative,error=error))
            summary=dict(id=identifier,kind="error",config_id=path.stem,path=relative,href=href,spec={},clusters=[],
                warnings=["转换失败"],comparison_group=None,probability_pass=None,audit_status="error")
            write_html(output/href,path.name,dict(summary=summary,error=error,record=None),"detail")
            summaries.append(summary);manifest.append(dict(path=relative,href=href,status="error",error=error))
        if i%25==0 or i==len(paths):print(f"  [{i}/{len(paths)}] exported; errors={len(errors)}",flush=True)
    models=[s for s in summaries if s["kind"]=="model"]
    best_groups=defaultdict(list)
    for s in models:
        if s.get("bic") is not None:best_groups[s["comparison_group"]].append(s["bic"])
    config_ll=defaultdict(list)
    for s in summaries:
        if s.get("ll") is not None:config_ll[s["config_id"]].append(s["ll"])
    for s in summaries:
        s["delta_bic"]=s["bic"]-min(best_groups[s["comparison_group"]]) if s.get("bic") is not None and s["comparison_group"] in best_groups else None
        s["delta_ll"]=max(config_ll[s["config_id"]])-s["ll"] if s.get("ll") is not None else None
    save_figures(output,summaries)
    progress=[]
    for path in sorted(input_dir.glob("test*/restarts/*.progress.json")):
        completed=path.with_name(path.name.replace(".progress.json",".pkl"))
        if not completed.exists():
            try:progress.append(dict(path=str(path.relative_to(input_dir)),**json.loads(path.read_text())))
            except (ValueError,OSError):pass
    catalog=dict(version=VERSION,generated_at=generated,input_dir=str(input_dir),files=summaries,
        counts=dict(total=len(paths),models=len(models),restarts=sum(s["kind"]=="restart" for s in summaries),
            errors=len(errors),tests=dict(Counter(s["spec"]["test"] for s in models)),progress_only=len(progress)),
        protocol=protocol,source_matches=bool(protocol) and not changed,source_differences=changed,
        audit_mode=args.audit,boundary_tolerance=args.boundary_tol,progress_only=progress)
    write_html(output/"index.html","Grid EM Conditional · 交互结果报告",catalog,"dashboard")
    (output/"catalog.json").write_text(json_text(catalog),encoding="utf-8")
    (output/"manifest.json").write_text(json.dumps(dict(version=VERSION,generated_at=generated,files=manifest,
        errors=errors,audit_mode=args.audit),ensure_ascii=False,indent=2),encoding="utf-8")
    import csv
    with (output/"parameters.csv").open("w",newline="",encoding="utf-8-sig") as fh:
        writer=csv.DictWriter(fh,fieldnames=["file","config_id","kind","cluster","pi","name","value","status","lower","upper","boundary"])
        writer.writeheader();writer.writerows(parameters)
    columns=["config_id","kind","path","bic","delta_bic","ll","delta_ll","k","n_obs","n_subjects","converged",
        "exit_reason","n_iter","probability_pass","near_zero_prelec","boundary_count","optimizer_unsuccessful",
        "optimizer_rejected","ll_gap","posterior_gap","normalized_rmse","ksi_median","posterior_median","href"]
    with (output/"diagnostics.csv").open("w",newline="",encoding="utf-8-sig") as fh:
        writer=csv.DictWriter(fh,fieldnames=columns,extrasaction="ignore");writer.writeheader();writer.writerows(summaries)
    (output/"README.md").write_text("# Grid EM Conditional 独立交互报告\n\n打开 index.html；details 和 assets 目录须保留在同一报告目录中。无需联网或启动服务器。\n\n"
        "总览默认展示模型结果；切换文件类型即可查看所有起点。每个 pickle 有独立详情页，原始字段、参数、个人 ξ、后验及完整迭代／优化器日志均可读取和下载。\n\n"
        "重新运行 python export_conditional_html.py 更新结果快照；运行中的估计不会被停止或修改。默认复算全部 pickle，--audit models 只复算模型，--audit none 只读取保存诊断。--models-only 仅导出模型 pickle。\n\n"
        "ΔBIC 只在相同测试、协议、参与者集合及观测数内比较；等价标签可隐藏。ΔLL 是同配置已导出起点距最高 LL 的差。单类后验为 1 由模型约束决定。模型边界、局部解与无标准误仍限制结构解释。\n",
        encoding="utf-8")
    print(json.dumps(dict(report=str(output/"index.html"),**catalog["counts"]),ensure_ascii=False),flush=True)
    return catalog


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir",type=Path,default=ROOT/"results_grid_conditional")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"reports"/"grid_em_conditional_explorer")
    parser.add_argument("--audit",choices=["all","models","none"],default="all")
    parser.add_argument("--models-only",action="store_true")
    parser.add_argument("--boundary-tol",type=float,default=1e-4)
    args=parser.parse_args()
    if args.boundary_tol<0:parser.error("--boundary-tol 必须非负")
    result=export(args)
    if result["counts"]["errors"]:sys.exit(1)


if __name__=="__main__":main()
