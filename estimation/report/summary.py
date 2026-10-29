"""Per-result report content: cluster parameters, saved diagnostics, the
scalar recomputation audit, and the catalog summary."""
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib

import numpy as np

from .. import cpt
from ..config import KSI_BOUNDS, VALUATION_VERSION
from ..diagnostics import probability_shape
from ..likelihood import component_loglik, count_free_params, posterior
from ..spec import CANON, bounds_free
from ..valuation import NAMES, named_parameters, reference_points, s_to_a, scalar_ce
from .i18n import localize
from .serialize import json_text, number, readable


def amplitude_zero_limit(lamb, alpha_gain, alpha_loss):
    """Right-hand limit of A(d)=lambda*d**(alpha_loss-alpha_gain), lambda>0."""
    if alpha_loss > alpha_gain:
        return 0.0
    if alpha_loss < alpha_gain:
        return float("inf")
    return float(lamb)


def utility_curve(values, extent=100):
    """Reference-centered power utility, sampled densely around zero."""
    positive = np.geomspace(extent*1e-6, extent, 201)
    offsets = np.concatenate((-positive[::-1], [0.0], positive))
    utility = np.array([cpt.u(z, R=0, alpha=values["alpha_gain"],
        alpha_loss=values["alpha_loss"], lamb=values["lambda"]) for z in offsets])
    return offsets, utility


def _boundary_side(value, bound, tolerance):
    if bound:
        if abs(value-bound[0]) <= tolerance:
            return "下界附近"
        if abs(value-bound[1]) <= tolerance:
            return "上界附近"
    return None


def cluster_parameters(result, tolerance, language="zh"):
    method = result["spec"].pw
    ix = CANON[method]
    resp = np.asarray(result["resp"], dtype=float)
    clusters = []
    for j, (theta, dle, layout, fv) in enumerate(zip(result["thetas"], result["delta_les"],
                                                   result["layouts"], result["free_vecs"])):
        values = named_parameters(theta, method)
        a1, a2, a3 = s_to_a(*(values[k] for k in ["s1", "s2", "s3"]))
        weights = [float(a1), float(a2), float(a3), float(1-a1-a2-a3)]
        bounds = bounds_free(layout)
        free_bounds = {pos: bounds[i] for i, pos in enumerate(layout.free_positions)}
        params = []
        for key, pos in ix.items():
            if key == "LEN":
                continue
            status = "自由估计" if pos in layout.free_positions else "固定"
            if key == "AL" and layout.link_AL_to_AG:
                status = "与 alpha_gain 绑定"
            if key == "DELTA" and weights[1]+weights[2] <= 1e-12:
                status += "；当前参考点未使用"
            bound = free_bounds.get(pos)
            value = float(theta[pos])
            params.append(dict(name=NAMES[key], value=value, status=status,
                lower=bound[0] if bound else None, upper=bound[1] if bound else None,
                boundary=_boundary_side(value, bound, tolerance)))
        effective_le = values["delta"] if dle is None else float(dle)
        le_bound = bounds[-1] if layout.has_delta_le else None
        params.append(dict(name="delta_le", value=effective_le,
            status=("自由估计" if layout.has_delta_le else "共享 delta" if dle is None else "固定")+
                   ("；当前参考点未使用" if weights[2] <= 1e-12 else ""),
            lower=le_bound[0] if le_bound else None, upper=le_bound[1] if le_bound else None,
            boundary=_boundary_side(effective_le, le_bound, tolerance)))
        parameter = values.get("gamma", values.get("palpha"))
        monotone, minimum = probability_shape(method, parameter)
        identity = result["rp_identities"][j]
        label = identity or (f"Class {j+1}" if language == "en" else f"第 {j+1} 类")
        clusters.append(dict(cluster=j+1, label=label, pi=float(result["pis"][j]),
            soft_n=float(resp[:, j].sum()), map_n=int(np.sum(resp.argmax(axis=1) == j)),
            parameters=params, values=values, weights=weights, effective_delta_le=effective_le,
            free_vec=readable(fv), method=method, probability_parameter=parameter,
            monotone=monotone, derivative_numerator_min=minimum,
            near_zero_prelec=method == "prelec" and parameter-1e-6 <= tolerance,
            boundary_count=sum(p["boundary"] is not None for p in params),
            amplitude_zero_limit=amplitude_zero_limit(values["lambda"], values["alpha_gain"], values["alpha_loss"])))
    for cluster in clusters:
        for parameter in cluster["parameters"]:
            for key in ["status", "boundary"]:
                parameter[key] = localize(parameter[key], language)
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
    return dict(ksi_quantiles=np.quantile(ksi, [0, .05, .25, .5, .75, .95, 1]).tolist(),
        ksi_at_lower=int(np.sum(np.abs(ksi-KSI_BOUNDS[0]) <= tolerance)),
        ksi_at_upper=int(np.sum(np.abs(ksi-KSI_BOUNDS[1]) <= tolerance)),
        ksi_bounds=list(KSI_BOUNDS), posterior_row_sum_gap=float(np.max(np.abs(resp.sum(axis=1)-1))),
        posterior_finite=bool(np.isfinite(resp).all()),
        posterior_in_unit_interval=bool(np.all((resp >= 0) & (resp <= 1))),
        posterior_median=float(np.median(maxpost)) if c > 1 else None,
        posterior_below_08=float(np.mean(maxpost < .8)) if c > 1 else None,
        normalized_entropy_mean=float(entropy.mean()/np.log(c)) if c > 1 else None,
        pi_posterior_gap=float(np.max(np.abs(np.asarray(result["pis"])-resp.mean(axis=0)))),
        ll_decreases=int(np.sum(changes < -1e-6)),
        last_logged_ll_change=float(np.diff(ll_values)[-1]) if len(ll_values) > 1 else None,
        final_recomputation_change=number(result.get("final_ll_change")),
        optimizer_calls=len(optimizer), optimizer_unsuccessful=sum(not r.get("success", False) for r in optimizer),
        optimizer_rejected=sum(not r.get("accepted", False) for r in optimizer),
        optimizer_q_increases=sum(number(r.get("q_after")) is not None and number(r.get("q_before")) is not None
            and r["q_after"] > r["q_before"]+1e-8 for r in optimizer),
        optimizer_messages=dict(Counter(str(r.get("message", "")) for r in optimizer)),
        n_subjects=n)


def audit_result(result, data, clusters):
    """Recompute CEs, likelihood and posteriors with the scalar CPT formula,
    including the actual decision weights of every conditional lottery."""
    if result["subjects"] != data.subjects:
        raise ValueError("参与者集合与当前数据不一致")
    method = result["spec"].pw
    state_mu = np.empty((len(data.states), len(clusters)))
    references = np.empty_like(state_mu)
    for j, (theta, dle, cluster) in enumerate(zip(result["thetas"], result["delta_les"], clusters)):
        rr = reference_points(theta, method, dle, data.reference_inputs)
        references[:, j] = rr
        v = cluster["values"]
        minimum = np.inf
        negative_states = 0
        for i, (_, current) in enumerate(data.states):
            state_mu[i, j], ev = scalar_ce(v, float(rr[i]), current, method)
            outcomes = defaultdict(float)
            for utility, p in zip(ev["present_values_of_streams"], ev["probabilities_of_streams"]):
                outcomes[utility] += p
            weights, _ = cpt.dw(outcomes, gamma=v.get("gamma", 1), beta=v.get("beta", 1),
                                palpha=v.get("palpha", 1), method=method)
            smallest = min(weights.values())
            minimum = min(minimum, smallest)
            negative_states += int(smallest < -1e-10)
        cluster.update(min_decision_weight=float(minimum), negative_states=negative_states)
    if not np.isfinite(state_mu).all():
        raise ValueError("理论 CE 出现非有限值")
    row_mu = state_mu[data.state_index]
    ll, resp = posterior(component_loglik(state_mu, np.asarray(result["ksi"]), data), result["pis"])
    residual = data.observed-np.sum(row_mu*resp[data.subject_index], axis=1)
    stages = data.reference_inputs[3][data.state_index]
    stage_stats = []
    for stage in [0, 1, 2]:
        mask = stages == stage
        stage_stats.append(dict(stage=stage, n=int(mask.sum()), bias=float(np.mean(residual[mask])),
            rmse=float(np.sqrt(np.mean(residual[mask]**2))),
            normalized_rmse=float(np.sqrt(np.mean((residual[mask]/data.spread[mask])**2)))))
    states = []
    for i, (key, _) in enumerate(data.states):
        mask = data.state_index == i
        states.append(dict(lottery=key[0], stage=int(key[1]), z1=float(key[2]), z2=float(key[3]), n=int(mask.sum()),
            observed_mean=float(data.observed[mask].mean()), conditional_expectation=float(data.reference_inputs[1][i]),
            reference=references[i].tolist(), ce=state_mu[i].tolist()))
    return dict(status="verified", ll_recomputed=ll, ll_gap=float(ll-result["log_likelihood"]),
        posterior_gap=float(np.max(np.abs(resp-result["resp"]))),
        normalized_rmse=float(np.sqrt(np.mean((residual/data.spread)**2))),
        rmse=float(np.sqrt(np.mean(residual**2))), stage=stage_stats, states=states)


def summarize(result, path, input_dir, meta, protocol, context, tolerance, do_audit, language="zh"):
    """(catalog summary, detail payload) of one saved result."""
    spec = asdict(result["spec"])
    clusters = cluster_parameters(result, tolerance, language)
    saved = saved_diagnostics(result, tolerance)
    version = result.get("valuation_version")
    fingerprint = result.get("run_fingerprint", meta.get("run_fingerprint", protocol.get("fingerprint")))
    fit = dict(status="not_run", reason="选择仅使用已保存诊断")
    if do_audit:
        if not context["data"]:
            fit = dict(status="not_run", reason=context["reason"])
        elif version != VALUATION_VERSION:
            fit = dict(status="not_run", reason="pickle 的估值版本不是当前条件剩余彩票版本")
        elif fingerprint != protocol.get("fingerprint"):
            fit = dict(status="not_run", reason="pickle 的协议指纹与目录协议不同")
        else:
            try:
                fit = audit_result(result, context["data"], clusters)
            except Exception as exc:
                fit = dict(status="error", reason=f"{type(exc).__name__}: {exc}")
    n_obs = int(result.get("n_obs_effective", result.get("n_obs", 0)))
    k = count_free_params(result)
    ll = float(result["log_likelihood"])
    bic = k*np.log(n_obs)-2*ll if n_obs > 0 else None
    kind = "model" if path.name == "result.pkl" else "restart"
    warnings = []
    if not result.get("converged"):
        warnings.append("未达到 LL 容差")
    if any(c["monotone"] is False for c in clusters):
        warnings.append("概率函数非单调")
    if any(c.get("negative_states", 0) > 0 for c in clusters):
        warnings.append("实际彩票有负决策权重")
    if any(c["near_zero_prelec"] for c in clusters):
        warnings.append("近零 Prelec 参数")
    if any(c["boundary_count"] for c in clusters):
        warnings.append("自由参数接近边界")
    if saved["optimizer_unsuccessful"]:
        warnings.append("存在优化器非成功退出")
    if saved["optimizer_rejected"]:
        warnings.append("存在被拒绝的 M 步")
    if saved["ll_decreases"]:
        warnings.append("记录中 LL 曾下降")
    if any(c["pi"] < .01 for c in clusters):
        warnings.append("存在小于 1% 的类别")
    if fit["status"] != "verified":
        warnings.append("未完成当前数据复算")
    elif abs(fit["ll_gap"]) > 1e-5:
        warnings.append("保存 LL 与复算不一致")
    probability_pass = all(c["monotone"] for c in clusters)
    if fit["status"] == "verified":
        probability_pass = probability_pass and all(c["negative_states"] == 0 for c in clusters)
    else:
        probability_pass = None if probability_pass else False
    subject_hash = hashlib.sha256(json_text(result["subjects"]).encode()).hexdigest()[:16]
    group = f"{fingerprint}|{version}|{n_obs}|{subject_hash}|{spec['test']}"
    brief = [{key: c.get(key) for key in ["cluster", "label", "pi", "soft_n", "map_n", "weights", "values",
        "method", "probability_parameter", "monotone", "near_zero_prelec", "boundary_count", "amplitude_zero_limit",
        "min_decision_weight", "negative_states"]} for c in clusters]
    summary = dict(config_id=result["spec"].config_id(), kind=kind, restart_id=result.get("restart_id"),
        path=str(path.relative_to(input_dir)), spec=spec, ll=ll, bic=number(bic), aic=2*k-2*ll, k=k,
        bic_saved=number(result.get("bic")), bic_gap=number(result.get("bic", bic)-bic) if bic is not None else None,
        n_obs=n_obs, n_subjects=len(result["subjects"]), converged=bool(result.get("converged")),
        exit_reason=result.get("exit_reason"), n_iter=result.get("n_iter"), elapsed=number(result.get("elapsed")),
        model_total_elapsed=number(meta.get("elapsed_seconds")) if kind == "model" else None,
        equivalent_to=result.get("equivalent_to", meta.get("equivalent_to")),
        probability_pass=probability_pass, near_zero_prelec=any(c["near_zero_prelec"] for c in clusters),
        boundary_count=sum(c["boundary_count"] for c in clusters), optimizer_unsuccessful=saved["optimizer_unsuccessful"],
        optimizer_rejected=saved["optimizer_rejected"], ll_decreases=saved["ll_decreases"],
        ksi_median=saved["ksi_quantiles"][3], posterior_median=saved["posterior_median"],
        audit_status=fit["status"], ll_gap=fit.get("ll_gap"), posterior_gap=fit.get("posterior_gap"),
        normalized_rmse=fit.get("normalized_rmse"), warnings=warnings, clusters=brief,
        comparison_group=group, fingerprint=fingerprint, valuation_version=version)
    summary["warnings"] = [localize(warning, language) for warning in warnings]
    if "reason" in fit:
        fit["reason"] = localize(fit["reason"], language)
    return summary, dict(summary=summary, clusters=clusters, diagnostics=saved, fit=fit,
        metadata=meta, record=readable(result))
