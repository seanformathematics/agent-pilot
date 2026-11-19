
import sys, json
import numpy as np
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]

def U_of(S2, S4, n): return 1.0 - (S4 / n) / (3.0 * (S2 / n) ** 2)
def jk(S2r, S4r, nper, G):
    R = S2r.size; k = R // G
    s2 = S2r.reshape(G, k).sum(1); s4 = S4r.reshape(G, k).sum(1)
    ng = nper * k; n = ng * G
    Uall = U_of(s2.sum(), s4.sum(), n)
    Uloo = U_of(s2.sum() - s2, s4.sum() - s4, n - ng)
    return Uall, G * Uall - (G - 1) * Uloo.mean(), np.sqrt((G - 1) / G * np.sum((Uloo - Uloo.mean()) ** 2))
def delta(S2r, S4r, nper):
    R = S2r.size; x2, x4 = S2r / nper, S4r / nper
    M2, M4 = x2.mean(), x4.mean()
    z = -x4 / (3 * M2 ** 2) + 2 * M4 * x2 / (3 * M2 ** 3)
    return np.sqrt(z.var(ddof=1) / R)

final = {}
hdr = f"{'T':>4} {'U (impr, pooled)':>17} | jk G=50  G=100  G=200  G=500 | delta  | blocks(20) | U_plain +- err"
print(hdr)
for T in Ts:
    d1 = np.load(f"/work/prod_T{T:.1f}.npz"); d2 = np.load(f"/work/prod2_T{T:.1f}.npz")
    blen = int(d1["blen"]); nb, R = d1["i2"].shape; nper = nb * blen
    I2 = np.concatenate([d1["i2"], d2["i2"]], axis=1); I4 = np.concatenate([d1["i4"], d2["i4"]], axis=1)   # (nb, 2R)
    P2 = np.concatenate([d1["p2"], d2["p2"]], axis=1); P4 = np.concatenate([d1["p4"], d2["p4"]], axis=1)
    S2, S4 = I2.sum(0), I4.sum(0)
    out = {G: jk(S2, S4, nper, G) for G in (50, 100, 200, 500)}
    Uall = out[200][0]
    dl = delta(S2, S4, nper)
    # time-block scatter: 20 blocks = 10 time-blocks x 2 independent runs; each block pools 1000 replicas x 200 steps
    Ub = []
    for dd in (d1, d2):
        for b in range(nb):
            Ub.append(U_of(dd["i2"][b].sum(), dd["i4"][b].sum(), R * blen))
    Ub = np.array(Ub)
    eblock = Ub.std(ddof=1) / np.sqrt(len(Ub))
    up, _, ep = jk(P2.sum(0), P4.sum(0), nper, 200)
    e_final = max(out[100][2], out[200][2], out[500][2], dl, eblock)
    final[T] = dict(U=Uall, U_bc=out[200][1], jk200=out[200][2], jk100=out[100][2], jk500=out[500][2], delta=dl, block=eblock,
                    Up=up, ep=ep, n=int(2 * R * nper))
    print(f"{T:4.1f} {Uall:11.6f} | {out[50][2]:.2e} {out[100][2]:.2e} {out[200][2]:.2e} {out[500][2]:.2e} | {dl:.2e} | {eblock:.2e} | {up:.6f} +- {ep:.2e}  (bc-U diff {out[200][1]-Uall:.1e})")
json.dump({str(k): v for k, v in final.items()}, open("/work/final_candidates.json", "w"), indent=1)
