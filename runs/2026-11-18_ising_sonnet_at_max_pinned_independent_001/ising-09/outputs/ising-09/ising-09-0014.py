
import sys, time, json
sys.path.insert(0, "/work")
import numpy as np
from sw import SW

def tau_int(x):
    """x: (R, n) time series from independent replicas. Sokal automatic windowing; returns tau_int (in sweeps)."""
    R, n = x.shape
    x = x - x.mean(axis=1, keepdims=True)    # (slightly biased: subtract replica means) -> use global mean instead
    xg = x
    var = (xg*xg).mean()
    # autocorrelation function averaged over replicas
    maxlag = 60
    C = np.array([ (xg[:, :n-t]*xg[:, t:]).mean() for t in range(maxlag)])/var
    tau = 0.5; 
    for W in range(1, maxlag):
        tau = 0.5 + C[1:W+1].sum()
        if W >= 6*tau: break
    return tau, C[:6]

for T in [1.5, 2.0, 2.269, 2.8]:
    sim = SW(16, 400, seed=5555+int(T*100), start="hot")
    for _ in range(200): sim.sweep(T)
    n = 600
    m2 = np.zeros((400, n)); m4 = np.zeros((400, n))
    for t in range(n):
        sim.sweep(T)
        m = sim.mag()
        m2[:, t] = m*m; m4[:, t] = m**4
    t2, C2 = tau_int(m2); t4, C4 = tau_int(m4)
    print(f"T={T}: tau_int(m^2)={t2:.2f}  tau_int(m^4)={t4:.2f} sweeps;  C_m2(1..5)={np.round(C2[1:6],3)}")

d = json.load(open("/work/final_summary.json"))
print()
for k, v in d.items():
    print(k, f"U={v['U']:.7f} err={v['err']:.7f} (blk {v['err_blk']:.7f}, improved: {v['Ui']:.7f} +- {v['erri']:.7f}) n={v['n']}")
