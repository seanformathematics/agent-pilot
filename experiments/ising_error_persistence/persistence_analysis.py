"""Persistence analysis for experiments/ising_error_persistence (design.md, commit 68e7856).

Read-only. Reads the run logs, never writes into runs/.

    python -m experiments.ising_error_persistence.persistence_analysis runs/<run> [runs/<run> ...] [--out DIR]

What it does, in the order of design.md:

1. Attempts. Every submit_result call of every agent, in order, classified as
   - "markup":    the call carries the `markup` label (harness/markup.py, version 1)
   - "malformed": rejected as an invalid submission, without markup
                  (missing_untraced, empty_input, unexpected_field, or any other validation error)
   - "sanity":    well formed, rejected by the sanity checks (or for NaN/infinite values)
   - "accepted":  accepted
   Calls made after the submission limit ("No submissions left") are not attempts.

2. Attempt error clusters (AECs). An AEC starts at a rejected "markup" attempt. It
   continues through "markup" and "malformed" attempts and ends at the first well-formed
   attempt ("sanity" or "accepted"). Its size is the number of failed attempts in it.
   An accepted attempt that still carries markup ends the AEC and is flagged.
   An AEC still open when the agent ends is censored, with the agent's end status as reason.
   Primary analysis: each agent's first AEC. Secondary: all AECs, and the alternative
   reading in which a "malformed" attempt also ends the AEC.

3. Fits, by maximum likelihood with censoring, of the AEC size L >= 1, written as the
   survival function S(k) = P(L > k):
   - geometric        S(k) = q^k                                   (1 parameter)
   - discrete Weibull S(k) = q^(k^beta)                            (2; geometric at beta = 1)
   - beta-geometric   S(k) = B(a, b + k) / B(a, b)                 (2; each AEC has its own
                      recovery chance r ~ Beta(a, b); geometric as a + b -> infinity)
   - zeta             S(k) = zeta(alpha, k + 1) / zeta(alpha, 1)   (1; descriptive only)
   An observed size k contributes S(k-1) - S(k); a censored one with k failures contributes S(k).
   (Design.md calls the Weibull shape k; here it is beta, to keep k for sizes.)

4. Decisions (design.md, "AEC Survival Analysis", item 2):
   - Shape: profile-likelihood interval of beta at 95% (chi2_1 = 3.841) and 3 sigma (chi2_1 = 9).
   - Heterogeneity: dAIC = AIC(geometric) - AIC(beta-geometric); 2 to 7 suggestive, 7+ evidence.
     The likelihood-ratio p-value is halved (boundary of the parameter space).

5. Kaplan-Meier curve of the first AECs with a Greenwood 95% band, the fitted survival
   curves overlaid; AEC rate = agents with a markup attempt / agents with any attempt.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, special, stats

from harness.markup import DETECTOR_VERSION, rows_for_log

CHI2_95 = stats.chi2.ppf(0.95, 1)     # 3.841
CHI2_3SIGMA = 9.0                      # (3 sigma)^2, two-sided p = 0.0027
DAIC_SUGGESTIVE, DAIC_EVIDENCE = 2.0, 7.0


# ----- 1. attempts -----------------------------------------------------------

def agent_attempts(log_path: Path, run_id: str) -> dict:
    """One agent's submit_result attempts, classified, plus how the agent ended."""
    entries = [json.loads(line) for line in log_path.read_text().splitlines()]
    messages = {e["data"]["tool_use_id"]: e["data"]["message"] for e in entries if e["kind"] == "rejection"}
    end = next((e["data"] for e in entries if e["kind"] == "agent_end"), {})
    attempts = []
    for row in rows_for_log(log_path, run_id):
        if row["tool"] != "submit_result":
            continue
        message = messages.get(row["tool_use_id"])
        if message is not None and message.startswith(("No submissions left", "A submission was already accepted")):
            continue
        has_markup = "markup" in row["issues"]
        if not row["rejected"]:
            kind = "accepted"
        elif has_markup:
            kind = "markup"
        elif message.startswith("Invalid submission:"):
            kind = "malformed"
        else:
            kind = "sanity"   # sanity checks, or NaN / infinite values: well formed, values rejected
        attempts.append({"seq": row["seq"], "kind": kind, "issues": row["issues"],
                         "accepted_with_markup": kind == "accepted" and has_markup})
    limit = entries[0]["data"]["config"]["limits"]["max_submissions"]
    status = end.get("status", "missing")
    # The agent loop reports an agent that used up its submissions as having ended its turn;
    # name the real reason so censoring is recorded correctly.
    if status != "submitted" and len(attempts) >= limit:
        status = "submission_limit"
    return {"run_id": run_id, "agent_id": log_path.parent.name, "end_status": status,
            "submission_limit": limit, "attempts": attempts}


# ----- 2. AECs ---------------------------------------------------------------

def find_aecs(attempts: list[dict], end_status: str, malformed_ends: bool = False) -> list[dict]:
    """Every AEC in one agent's attempts. size = failed attempts; censored if still open at the end."""
    aecs, current = [], None
    for index, attempt in enumerate(attempts, start=1):
        kind = attempt["kind"]
        if current is None:
            if kind == "markup":
                current = {"start_attempt": index, "size": 1, "labels": [attempt["issues"]]}
            continue
        failed = kind == "markup" or (kind == "malformed" and not malformed_ends)
        if failed:
            current["size"] += 1
            current["labels"].append(attempt["issues"])
        else:
            current.update(censored=False, ended_by=kind, reason=None,
                           accepted_with_markup=attempt["accepted_with_markup"])
            aecs.append(current)
            current = None
    if current is not None:
        current.update(censored=True, ended_by=None, reason=end_status, accepted_with_markup=False)
        aecs.append(current)
    return aecs


# ----- 3. models -------------------------------------------------------------
# Each model: (name, number of parameters, survival(k, theta), starting theta).
# theta lives on an unconstrained scale; the functions map it to the parameters.

def _q(x):
    return 1.0 / (1.0 + math.exp(-x))


def surv_geometric(k, t):
    return _q(t[0]) ** k


def surv_weibull(k, t):
    return _q(t[0]) ** (k ** math.exp(t[1]))


def surv_betageom(k, t):
    a, b = math.exp(t[0]), math.exp(t[1])
    return math.exp(special.betaln(a, b + k) - special.betaln(a, b))


def surv_zeta(k, t):
    alpha = 1.0 + math.exp(t[0])
    return special.zeta(alpha, k + 1) / special.zeta(alpha, 1)


MODELS = {
    "geometric": (1, surv_geometric, [0.4]),
    "weibull": (2, surv_weibull, [0.4, 0.0]),
    "beta_geometric": (2, surv_betageom, [0.0, 0.5]),
    "zeta": (1, surv_zeta, [0.0]),
}


def neg_loglik(theta, surv, data):
    total = 0.0
    for size, censored in data:
        p = surv(size, theta) if censored else surv(size - 1, theta) - surv(size, theta)
        total += math.log(max(p, 1e-300))
    return -total


def fit(name, data, fixed_beta=None):
    n_par, surv, start = MODELS[name]
    if fixed_beta is not None:   # profile likelihood of the Weibull shape
        objective = lambda t: neg_loglik([t[0], math.log(fixed_beta)], surv_weibull, data)
        start = [0.4]
    else:
        objective = lambda t: neg_loglik(t, surv, data)
    best = None
    for shift in (0.0, 1.0, -1.0, 2.0):   # a few starts; keep the best
        result = optimize.minimize(objective, [s + shift for s in start], method="Nelder-Mead",
                                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000})
        if best is None or result.fun < best.fun:
            best = result
    return best


def parameters(name, theta):
    if name == "geometric":
        return {"q_continue": _q(theta[0])}
    if name == "weibull":
        return {"q": _q(theta[0]), "beta": math.exp(theta[1])}
    if name == "beta_geometric":
        a, b = math.exp(theta[0]), math.exp(theta[1])
        return {"a": a, "b": b, "mean_recovery": a / (a + b), "concentration": a + b}
    return {"alpha": 1.0 + math.exp(theta[0])}


def beta_interval(data, best_nll, threshold):
    """Profile-likelihood interval of the Weibull shape: beta where 2*(nll - best) <= threshold."""
    def excess(log_beta):
        return 2.0 * (fit("weibull", data, fixed_beta=math.exp(log_beta)).fun - best_nll) - threshold

    best_log_beta = math.log(parameters("weibull", fit("weibull", data).x)["beta"])
    bounds = []
    for direction in (-1, 1):
        step, edge = 0.05, best_log_beta
        while excess(edge + direction * step) < 0 and abs(edge - best_log_beta) < 6:
            edge += direction * step
        far = edge + direction * step
        bounds.append(math.exp(optimize.brentq(excess, edge, far)) if excess(far) > 0 else None)
    return bounds   # [low, high]; None = unbounded within the search range


def model_comparison(data):
    fits = {name: fit(name, data) for name in MODELS}
    out = {}
    for name, result in fits.items():
        n_par = MODELS[name][0]
        out[name] = {"parameters": parameters(name, result.x), "loglik": -result.fun,
                     "n_parameters": n_par, "aic": 2 * n_par + 2 * result.fun}
    best_aic = min(m["aic"] for m in out.values())
    weights = {n: math.exp(-(m["aic"] - best_aic) / 2) for n, m in out.items()}
    total = sum(weights.values())
    for name, m in out.items():
        m["delta_aic"] = m["aic"] - best_aic
        m["akaike_weight"] = weights[name] / total
    return out, fits


# ----- 4. decisions ----------------------------------------------------------

def decisions(data, comparison, fits):
    low95, high95 = beta_interval(data, fits["weibull"].fun, CHI2_95)
    low3, high3 = beta_interval(data, fits["weibull"].fun, CHI2_3SIGMA)
    beta = comparison["weibull"]["parameters"]["beta"]

    def side(low, high):
        if low is not None and low > 1:
            return "above"
        if high is not None and high < 1:
            return "below"
        return None

    s3, s95 = side(low3, high3), side(low95, high95)
    if s3 or s95:
        tier = "evidence (≈3σ)" if s3 else "suggestive"
        reading = ("k > 1: recovery gets likelier (H3, learning)" if (s3 or s95) == "above"
                   else "k < 1: recovery gets less likely (H2, or failure feeding on itself; not separable)")
        shape_verdict = f"{tier}: {reading}"
    else:
        shape_verdict = "consistent with H1 (interval contains 1)"

    d_aic = comparison["geometric"]["aic"] - comparison["beta_geometric"]["aic"]
    lr = 2 * (comparison["beta_geometric"]["loglik"] - comparison["geometric"]["loglik"])
    p_lr = 0.5 * stats.chi2.sf(max(lr, 0.0), 1)
    if d_aic >= DAIC_EVIDENCE:
        het_verdict = "evidence (≈3σ) that recovery chances differ between agents (H2)"
    elif d_aic >= DAIC_SUGGESTIVE:
        het_verdict = "suggestive that recovery chances differ between agents (H2)"
    else:
        het_verdict = "inconclusive"
    return {
        "shape": {"beta": beta, "interval_95": [low95, high95], "interval_3sigma": [low3, high3],
                  "width_95": (high95 - low95) if (low95 and high95) else None, "verdict": shape_verdict},
        "heterogeneity": {"delta_aic_geometric_minus_betageometric": d_aic, "lr_statistic": lr,
                          "p_value_halved": p_lr, "verdict": het_verdict},
    }


# ----- 5. Kaplan-Meier -------------------------------------------------------

def kaplan_meier(data):
    """S(k) = P(L > k) for k = 0..max, with a Greenwood 95% band (log-log transformed)."""
    max_k = max(size for size, _ in data)
    rows, s, var_sum = [{"k": 0, "S": 1.0, "low": 1.0, "high": 1.0, "at_risk": len(data)}], 1.0, 0.0
    for k in range(1, max_k + 1):
        at_risk = sum(1 for size, _ in data if size >= k)
        recovered = sum(1 for size, c in data if size == k and not c)
        if at_risk and recovered:
            s *= 1 - recovered / at_risk
            if at_risk > recovered:
                var_sum += recovered / (at_risk * (at_risk - recovered))
        if 0 < s < 1 and var_sum > 0:
            half = 1.96 * math.sqrt(var_sum) / abs(math.log(s))
            low, high = s ** math.exp(half), s ** math.exp(-half)
        else:
            low = high = s
        rows.append({"k": k, "S": s, "low": low, "high": high, "at_risk": at_risk})
    return rows


def plot(km, fits, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = [r["k"] for r in km]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.step(ks, [r["S"] for r in km], where="post", color="black", label="Kaplan–Meier (first AECs)")
    ax.fill_between(ks, [r["low"] for r in km], [r["high"] for r in km], step="post", color="0.85",
                    label="95% band (Greenwood)")
    grid = np.arange(0, max(ks) + 1)
    for name, style in [("geometric", "-"), ("weibull", "--"), ("beta_geometric", "-."), ("zeta", ":")]:
        surv = MODELS[name][1]
        ax.plot(grid, [surv(k, fits[name].x) for k in grid], style, marker="o", markersize=3,
                drawstyle="steps-post", label=name)   # sizes are whole numbers
    ax.set_xlabel("failed attempts k")
    ax.set_xticks(grid)
    ax.set_ylabel("P(AEC size > k)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----- driver ----------------------------------------------------------------

def analyse_aecs(first_aecs):
    data = [(a["size"], a["censored"]) for a in first_aecs]
    if len(data) < 3 or all(c for _, c in data):
        return {"n": len(data), "note": "too few uncensored AECs to fit"}
    comparison, fits = model_comparison(data)
    return {"n": len(data), "n_censored": sum(c for _, c in data), "models": comparison,
            "decisions": decisions(data, comparison, fits), "_fits": fits, "_data": data}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", default=None, help="folder for result.json and km.png (never inside runs/)")
    args = parser.parse_args(argv)

    agents = []
    for run in args.runs:
        run_dir = Path(run)
        for log_path in sorted(run_dir.glob("**/log.jsonl")):
            agents.append(agent_attempts(log_path, run_dir.name))

    with_attempts = [a for a in agents if a["attempts"]]
    with_markup = [a for a in with_attempts if any(x["kind"] == "markup" for x in a["attempts"])]
    per_agent = []
    for a in agents:
        main_aecs = find_aecs(a["attempts"], a["end_status"])
        alt_aecs = find_aecs(a["attempts"], a["end_status"], malformed_ends=True)
        per_agent.append({**{k: a[k] for k in ("run_id", "agent_id", "end_status", "submission_limit")},
                          "attempt_kinds": [x["kind"] for x in a["attempts"]],
                          "aecs": main_aecs, "aecs_alternative": alt_aecs})

    primary = analyse_aecs([p["aecs"][0] for p in per_agent if p["aecs"]])
    alternative = analyse_aecs([p["aecs_alternative"][0] for p in per_agent if p["aecs_alternative"]])
    all_aecs = [aec for p in per_agent for aec in p["aecs"]]

    result = {
        "detector_version": DETECTOR_VERSION,
        "runs": [Path(r).name for r in args.runs],
        "agents": len(agents),
        "aec_rate": {"agents_with_markup_attempt": len(with_markup), "agents_with_any_attempt": len(with_attempts)},
        "primary_first_aecs": {k: v for k, v in primary.items() if not k.startswith("_")},
        "secondary_alternative_reading": {k: v for k, v in alternative.items() if not k.startswith("_")},
        "secondary_all_aecs": {"n": len(all_aecs),
                               "agents_with_more_than_one": sum(1 for p in per_agent if len(p["aecs"]) > 1),
                               "sizes": sorted((a["size"], a["censored"]) for a in all_aecs)},
        "censoring_reasons": sorted({a["reason"] for p in per_agent for a in p["aecs"][:1] if a["censored"]}),
        "accepted_with_markup": sum(a["accepted_with_markup"] for a in all_aecs),
        "per_agent": per_agent,
    }
    if "_data" in primary:
        result["kaplan_meier_first_aecs"] = kaplan_meier(primary["_data"])

    if args.out:
        out = Path(args.out).resolve()
        if "runs" in out.parts:
            sys.exit("refusing to write inside runs/")
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
        if "_fits" in primary:
            plot(result["kaplan_meier_first_aecs"], primary["_fits"], out / "km.png")
    summary = {k: result[k] for k in ("agents", "aec_rate", "accepted_with_markup", "censoring_reasons")}
    summary["primary"] = result["primary_first_aecs"]
    summary["alternative"] = {k: v for k, v in result["secondary_alternative_reading"].items() if k == "decisions"}
    summary["all_aecs"] = {k: v for k, v in result["secondary_all_aecs"].items() if k != "sizes"}
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
