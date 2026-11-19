
import sys, json, time
sys.path.insert(0, '/work')
import numpy as np
import sw, prod

TS = prod.TS
rng = np.random.default_rng(555)

def tau_int(x):
    x = x - x.mean()
    var = (x * x).mean()
    tau = 0.5
    for lag in range(1, 300):
        c = (x[:-lag] * x[lag:]).mean() / var
        tau += c
        if lag >= 6 * tau:
            break
    return tau

final = []
t0 = time.time()
for i, T in enumerate(TS):
    d = np.load(f'/work/prod_T{T}.npz')
    R = int(d['R']); nmeas = int(d['nmeas']); neq = int(d['neq'])
    a = d['m2'].mean(axis=1); b = d['m4'].mean(axis=1)
    ai = d['m2i'].mean(axis=1); bi = d['m4i'].mean(axis=1)
    U, ejk = sw.jackknife_U(a, b, 100)
    Ui, eji = sw.jackknife_U(ai, bi, 100)
    # bootstrap over replicas
    Ub = np.empty(2000)
    for k in range(2000):
        idx = rng.integers(0, R, size=R)
        Ub[k] = sw.binder(a[idx].mean(), b[idx].mean())
    eboot = Ub.std(ddof=1)
    # short pilot for tau_int of m^2 (stationary, separate seed)
    sim = sw.SwendsenWang(16, 200, T, 5000 + i, start="hot")
    for _ in range(300):
        sim.sweep(measure=False)
    ts = np.empty((1000, 200))
    for it in range(1000):
        ts[it] = sim.sweep(measure=True)["m2"]
    tau = tau_int(ts)
    final.append(dict(T=T, U=float(U), U_err_jk=float(ejk), U_err_boot=float(eboot), U_impr=float(Ui), U_impr_err=float(eji),
                      n_samples=R * nmeas, R=R, nmeas=nmeas, neq=neq, tau_int_m2=float(tau)))
    print(f"T={T}: U={U:.7f}  jk={ejk:.3e}  boot={eboot:.3e}  | improved {Ui:.7f}+-{eji:.1e} (diff z={(U-Ui)/np.sqrt(max(ejk**2-eji**2,1e-30)):+.2f}) | n={R*nmeas} tau_int(m2)~{tau:.2f}", flush=True)
with open('/work/final_results.json', 'w') as f:
    json.dump(final, f, indent=1)
print("time", time.time() - t0)
print("max jk err:", max(r['U_err_jk'] for r in final))
