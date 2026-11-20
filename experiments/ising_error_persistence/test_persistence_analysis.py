import math, random
import persistence_analysis as pa

def att(*kinds):
    return [{"kind": k, "issues": ["markup"] if k == "markup" else [], "accepted_with_markup": False} for k in kinds]

def sizes(kinds, status="submitted", **kw):
    return [(a["size"], a["censored"], a["ended_by"]) for a in pa.find_aecs(att(*kinds), status, **kw)]

def test_aec_rules_from_design():
    assert sizes(["accepted"]) == []                                         # no markup, no AEC
    assert sizes(["markup", "accepted"]) == [(1, False, "accepted")]
    assert sizes(["markup", "markup", "malformed", "accepted"]) == [(3, False, "accepted")]  # malformed continues
    assert sizes(["markup", "markup", "malformed", "accepted"], malformed_ends=True) == [(2, False, "malformed")]
    assert sizes(["markup", "sanity", "accepted"]) == [(1, False, "sanity")]  # sanity rejection ends it
    assert sizes(["malformed", "accepted"]) == []                           # malformed alone never starts one
    assert sizes(["sanity", "malformed", "accepted"]) == []
    assert sizes(["markup", "sanity", "markup", "markup", "accepted"]) == [(1, False, "sanity"), (2, False, "accepted")]
    assert sizes(["markup"] * 3, status="submission_limit") == [(3, True, None)]

def test_fits_recover_known_shapes():
    random.seed(3)
    for beta in (0.6, 1.0, 1.6):
        q, data = 0.6, []
        for _ in range(3000):
            k = 1
            while k < 20 and random.random() < (q ** (k ** beta)) / (q ** ((k - 1) ** beta)):
                k += 1
            data.append((k, k == 20 and random.random() < q ** (20 ** beta) / q ** (19 ** beta)))
        comparison, fits = pa.model_comparison(data)
        got = comparison["weibull"]["parameters"]
        assert abs(got["beta"] - beta) < 0.08 and abs(got["q"] - q) < 0.03, (beta, got)

def test_survival_functions_are_proper():
    for name, (n, surv, start) in pa.MODELS.items():
        assert abs(surv(0, start) - 1) < 1e-12
        assert all(surv(k, start) >= surv(k + 1, start) for k in range(30)), name

def test_censored_geometric_closed_form():
    # 3 recovered after 1, 2, 3 failures; 2 censored after 3 failures. Geometric MLE of q = failures beyond the first / exposure.
    data = [(1, False), (2, False), (3, False), (3, True), (3, True)]
    comparison, _ = pa.model_comparison(data)
    continuations = sum(k - 1 for k, _ in data) + sum(1 for k, c in data if c)   # times the AEC went on
    trials = continuations + sum(1 for _, c in data if not c)                     # plus recoveries
    assert abs(comparison["geometric"]["parameters"]["q_continue"] - continuations / trials) < 1e-4
