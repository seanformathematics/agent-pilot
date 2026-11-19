
import sys, json
sys.path.insert(0, "/work")
import numpy as np
import sw

TS = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
NT = 1000
def load(tag, T):
    d = np.load(f"/work/{tag}_T{T:.1f}.npz")
    return {k: d[k] for k in d.files}

final = []
print("T     U (pooled improved)    U_err (jackknife)    U_plain+-err (no cluster-improvement)   n_samples")
for T in TS:
    d1, d2 = load("run1", T), load("run2", T)
    S2 = np.concatenate([d1["S2i"].sum(1), d2["S2i"].sum(1)])
    S4 = np.concatenate([d1["S4i"].sum(1), d2["S4i"].sum(1)])
    P2 = np.concatenate([d1["S2p"].sum(1), d2["S2p"].sum(1)])
    P4 = np.concatenate([d1["S4p"].sum(1), d2["S4p"].sum(1)])
    U, e, Ubc = sw.jackknife_U(S2, S4, NT)
    Up, ep, _ = sw.jackknife_U(P2, P4, NT)
    nsamp = int(len(S2) * NT)
    final.append(dict(T=T, U=float(U), U_err=float(e), U_jack_biascorr=float(Ubc), U_plain=float(Up), U_plain_err=float(ep), n_samples=nsamp, n_replicas=int(len(S2))))
    print(f"{T:.1f}   {U:.7f}   {e:.2e}   {Up:.6f}+-{ep:.2e}   {nsamp}")
json.dump(final, open("/work/final_results.json", "w"), indent=1)
print("max U_err =", max(f["U_err"] for f in final))

# ---- integrated autocorrelation time of the linearised fluctuation of U (improved estimators) ----
def tau_int(g):
    # g: (R, n) series; replica-averaged autocovariance, Madras-Sokal windowing
    R, n = g.shape
    g = g - g.mean()
    C = np.array([(g[:, : n - t] * g[:, t:]).mean() for t in range(0, 120)])
    tau = 0.5; 
    for W in range(1, 120):
        tau = 0.5 + C[1:W + 1].sum() / C[0]
        if W >= 6 * tau:
            break
    return tau

print("\nIntegrated autocorrelation time (sweeps) of SW for g = m4_imp - 2(A/B) m2_imp (drives U fluctuations):")
for T in [1.5, 2.0, 2.3, 2.6, 2.8]:
    rng = np.random.default_rng(777 + int(T * 10))
    sim = sw.SW(16, 300, T, rng)
    s = sim.init_state("cold")
    m2s, m4s = [], []
    for t in range(100 + 600):
        s, a, b = sim.step(s)
        if t >= 100:
            m2s.append(a.copy()); m4s.append(b.copy())
    m2s = np.array(m2s).T; m4s = np.array(m4s).T   # (R, n)
    A, B = m4s.mean(), m2s.mean()
    g = m4s - 2 * (A / B) * m2s
    print(f"  T={T:.1f}: tau_int(g) = {tau_int(g):.2f}   tau_int(m2_imp) = {tau_int(m2s):.2f}")
