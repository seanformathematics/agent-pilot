
import sys, json, time
sys.path.insert(0, '/work')
import numpy as np
import sw, prod

TS = prod.TS
def delta_err(a, b):
    R = len(a)
    A = a.mean(); B = b.mean()
    dA = 2 * B / (3 * A ** 3); dB = -1.0 / (3 * A ** 2)
    IF = dA * (a - A) + dB * (b - B)
    return np.sqrt(np.sum(IF ** 2) / (R * (R - 1)))

final = json.load(open('/work/final_results.json'))
for r in final:
    d = np.load(f"/work/prod_T{r['T']}.npz")
    a = d['m2'].mean(axis=1); b = d['m4'].mean(axis=1)
    r['U_err_delta'] = float(delta_err(a, b))
    r['U_err'] = float(max(r['U_err_jk'], r['U_err_boot'], r['U_err_delta']))
    print(f"T={r['T']}: jk={r['U_err_jk']:.3e} boot={r['U_err_boot']:.3e} delta={r['U_err_delta']:.3e} -> U_err={r['U_err']:.3e}")
json.dump(final, open('/work/final_results.json', 'w'), indent=1)

# ---- direct replication of the whole procedure with fresh seeds ----
t0 = time.time()
for T, seed in [(2.1, 8101), (2.3, 8102), (2.6, 8103), (2.8, 8104)]:
    R = 1000
    sim = sw.SwendsenWang(16, R, T, seed, start="hot")
    sim.s[:R // 2] = 1
    for _ in range(300):
        sim.sweep(measure=False)
    a = np.zeros(R); b = np.zeros(R)
    for _ in range(2000):
        o = sim.sweep(measure=True)
        a += o["m2"]; b += o["m4"]
    a /= 2000; b /= 2000
    U2, ej2 = sw.jackknife_U(a, b, 100)
    ed2 = delta_err(a, b)
    err2 = max(ej2, ed2)
    r = [x for x in final if x['T'] == T][0]
    z = (U2 - r['U']) / np.hypot(err2, r['U_err'])
    print(f"T={T}: production U={r['U']:.6f}+-{r['U_err']:.2e}; replicate U={U2:.6f}+-{err2:.2e}; z={z:+.2f}   [{time.time()-t0:.0f}s]", flush=True)
