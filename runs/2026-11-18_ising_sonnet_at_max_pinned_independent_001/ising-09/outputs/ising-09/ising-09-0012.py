
import sys, time, json
sys.path.insert(0, "/work")
import numpy as np
from run import ratio_U, jackknife_U
from exact import energy_exact

Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
rng = np.random.default_rng(424242)
metro = np.load("/work/metro_res.npz")
final = {}
print("T   |  U_plain  err_rep  err_blk  err_boot | U_impr  err | diff(p-i) +- jk | hot-cold z | 1st-2nd half z | e z | n")
for T in Ts:
    d = np.load(f"/work/prod_T{T:.1f}.npz")
    hot, cold = d["hot"], d["cold"]            # (5, R, nb)
    acc = np.concatenate([hot, cold], axis=1)
    _, R, nb = acc.shape
    L = int(d["L"]); S = int(d["S"])
    n_samples = R*nb*L
    # plain, replica-level jackknife
    rep2 = acc[0].mean(axis=1); rep4 = acc[1].mean(axis=1)
    U, err_rep, Ujk = jackknife_U(rep2, rep4)
    # block-level jackknife
    _, err_blk, _ = jackknife_U(acc[0].ravel(), acc[1].ravel())
    # bootstrap over replicas
    B = 1000
    idx = rng.integers(0, R, size=(B, R))
    A2 = rep2[idx].mean(axis=1); A4 = rep4[idx].mean(axis=1)
    Ub = ratio_U(A2, A4)
    err_boot = Ub.std(ddof=1)
    # improved
    irep2 = acc[2].mean(axis=1); irep4 = acc[3].mean(axis=1)
    Ui, erri, _ = jackknife_U(irep2, irep4)
    # jackknife of the difference plain-improved
    n = R
    def Uvec(x2, x4):
        A2_i = (x2.sum() - x2)/(n-1); A4_i = (x4.sum() - x4)/(n-1)
        return ratio_U(A2_i, A4_i)
    dvec = Uvec(rep2, rep4) - Uvec(irep2, irep4)
    dd = (n-1)/n*np.sum((dvec - dvec.mean())**2)
    diff = U - Ui; diff_err = np.sqrt(dd)
    # hot vs cold
    Rh = hot.shape[1]
    Uh, eh, _ = jackknife_U(hot[0].mean(axis=1), hot[1].mean(axis=1))
    Uc, ec, _ = jackknife_U(cold[0].mean(axis=1), cold[1].mean(axis=1))
    zhc = (Uh-Uc)/np.sqrt(eh**2+ec**2)
    # first vs second half in time
    h = nb//2
    U1, e1, _ = jackknife_U(acc[0][:, :h].mean(axis=1), acc[1][:, :h].mean(axis=1))
    U2, e2, _ = jackknife_U(acc[0][:, h:].mean(axis=1), acc[1][:, h:].mean(axis=1))
    # NB: the two halves use the same replicas, so they are (almost) independent in time but share replicas; use sqrt(e1^2+e2^2) as a conservative z
    zhalf = (U1-U2)/np.sqrt(e1**2+e2**2)
    # energy
    rep_e = acc[4].mean(axis=1)
    e_mc = rep_e.mean(); e_err = rep_e.std(ddof=1)/np.sqrt(R)
    ze = (e_mc - energy_exact(16, T))/e_err
    print(f"{T:.1f} | {U:.5f}  {err_rep:.5f}  {err_blk:.5f}  {err_boot:.5f} | {Ui:.5f} {erri:.5f} | {diff:+.5f} +- {diff_err:.5f} | {zhc:+.2f} | {zhalf:+.2f} | {ze:+.2f} | {n_samples}")
    final[T] = dict(U=U, err=err_rep, err_blk=err_blk, err_boot=err_boot, Ui=Ui, erri=erri, n=n_samples, Ujk=Ujk)
json.dump({str(k): v for k, v in final.items()}, open("/work/final_summary.json","w"), indent=1)
print()
for T in [2.0, 2.3, 2.6]:
    Um, em = metro[f"T{T}"]
    f = final[T]
    z = (f['U']-Um)/np.sqrt(f['err']**2+em**2); zi = (f['Ui']-Um)/np.sqrt(f['erri']**2+em**2)
    print(f"T={T}: SW plain {f['U']:.5f}+-{f['err']:.5f}, SW improved {f['Ui']:.5f}+-{f['erri']:.5f}  vs Metropolis {Um:.5f}+-{em:.5f}:  z(plain)={z:+.2f} z(impr)={zi:+.2f}")
