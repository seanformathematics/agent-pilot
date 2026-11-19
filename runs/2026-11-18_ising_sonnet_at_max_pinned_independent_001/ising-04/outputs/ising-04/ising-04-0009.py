
import sys, time, json
import numpy as np
sys.path.insert(0, "/work")
from prod import run_T

def U_of(S2, S4, n):
    return 1.0 - (S4 / n) / (3.0 * (S2 / n) ** 2)

def jk(S2r, S4r, nper, G):
    R = S2r.size; k = R // G
    s2 = S2r.reshape(G, k).sum(1); s4 = S4r.reshape(G, k).sum(1)
    ng = nper * k; n = ng * G
    Uall = U_of(s2.sum(), s4.sum(), n)
    Uloo = U_of(s2.sum() - s2, s4.sum() - s4, n - ng)
    return Uall, np.sqrt((G - 1) / G * np.sum((Uloo - Uloo.mean()) ** 2))

main = json.load(open("/work/results_main.json"))
Ts = [1.6, 2.0, 2.3, 2.5, 2.8]
NREP, R, G = 20, 500, 50
rep_log = {}
t00 = time.time()
for T in Ts:
    Ui, Up, Ei, Ep, starts = [], [], [], [], []
    for j in range(NREP):
        seed = 200000 + 1000 * int(round(T * 10)) + j
        start = "hot" if j % 2 == 0 else "cold"
        out, blen = run_T(T, seed, R=R, neq=100, nblocks=1, blen=200, start=start)
        u, e = jk(out["i2"][0], out["i4"][0], blen, G); Ui.append(u); Ei.append(e)
        u, e = jk(out["p2"][0], out["p4"][0], blen, G); Up.append(u); Ep.append(e)
        starts.append(start)
    Ui, Up, Ei, Ep = map(np.array, (Ui, Up, Ei, Ep))
    hot = np.array(starts) == "hot"
    ref = main[str(T)]
    for name, U, E, refU, referr in (("impr", Ui, Ei, ref["U"], ref["err"]), ("plain", Up, Ep, ref["Up"], ref["ep"])):
        sc = U.std(ddof=1)
        zmean = (U.mean() - refU) / np.sqrt(sc ** 2 / NREP + referr ** 2)
        zhc = (U[hot].mean() - U[~hot].mean()) / (sc * np.sqrt(2 / (NREP / 2)))
        print(f"T={T} {name:5s}: repeat-mean U={U.mean():.5f} scatter={sc:.5f} mean jk-err={E.mean():.5f} ratio={sc/E.mean():.2f} | prod U={refU:.5f} z(mean-prod)={zmean:5.2f} | hot-cold z={zhc:5.2f}")
    rep_log[T] = dict(Ui=Ui.tolist(), Ei=Ei.tolist(), Up=Up.tolist(), Ep=Ep.tolist())
    print(f"   (elapsed {time.time()-t00:.0f}s)", flush=True)
json.dump({str(k): v for k, v in rep_log.items()}, open("/work/calib_repeats.json", "w"))
