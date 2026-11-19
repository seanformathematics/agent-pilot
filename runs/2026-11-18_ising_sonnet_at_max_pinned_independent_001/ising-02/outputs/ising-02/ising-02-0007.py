
import sys, json
sys.path.insert(0,"/work")
import numpy as np
from analysis import *

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
metro = json.load(open("/work/metropolis_validation.json"))
summary = {}
print("T     U          err_blkJK  err_repJK  err_delta  tau_u(flips) | Ucold      Uhot       z_hc  | U_1sthalf U_2ndhalf z_halves | n_total")
for T in Ts:
    d = np.load(f"/work/wolff_T{T}.npz")
    M = d["M"]; hot = d["hot"].astype(bool)
    R, nm = M.shape
    m2, m4 = moments(M.reshape(-1))
    m2 = m2.reshape(R, nm); m4 = m4.reshape(R, nm)
    # per-replica sums
    S2r = m2.sum(1); S4r = m4.sum(1); cnt_r = np.full(R, nm)
    U_all, err_rep, Ubc_rep = jackknife_U(S2r, S4r, cnt_r)
    # block jackknife: B blocks per replica
    B = 10 if nm >= 20000 else 5
    L = nm//B
    S2b = m2[:, :L*B].reshape(R, B, L).sum(2).reshape(-1); S4b = m4[:, :L*B].reshape(R, B, L).sum(2).reshape(-1)
    cntb = np.full(R*B, L)
    U_blk, err_blk, Ubc_blk = jackknife_U(S2b, S4b, cntb)
    # delta method w/ Sokal windowing on each replica (influence series built from pooled means)
    a = m2.mean(); b = m4.mean()
    taus = []; var_means = []
    for r in range(R):
        u = -(m4[r]-b)/(3*a*a) + (2*b/(3*a**3))*(m2[r]-a)
        tau, W = tau_int_sokal(u)
        taus.append(tau)
        var_means.append(u.var()*tau/nm)
    err_delta = np.sqrt(np.sum(var_means))/R   # independent replicas: var of mean of R means
    tau_u = np.mean(taus)
    # hot vs cold
    def Usub(mask):
        _, e, _ = jackknife_U(S2r[mask], S4r[mask], cnt_r[mask])
        return U_of(S2r[mask].sum()/cnt_r[mask].sum(), S4r[mask].sum()/cnt_r[mask].sum()), e
    Uc, ec = Usub(~hot); Uh, eh = Usub(hot)
    z_hc = (Uc-Uh)/np.sqrt(ec**2+eh**2)
    # halves
    h = nm//2
    S2a = m2[:, :h].sum(1); S4a = m4[:, :h].sum(1); S2c = m2[:, h:2*h].sum(1); S4c = m4[:, h:2*h].sum(1)
    Ua, ea, _ = jackknife_U(S2a, S4a, np.full(R, h)); Ub, eb, _ = jackknife_U(S2c, S4c, np.full(R, h))
    z_half = (Ua-Ub)/np.sqrt(ea**2+eb**2)
    summary[T] = dict(U=U_all, err_blk=err_blk, err_rep=err_rep, err_delta=err_delta, n=R*nm, tau=tau_u, Ubc=Ubc_blk,
                      z_hc=z_hc, z_half=z_half, cbar=float(d["cbar"].mean()))
    print(f"{T:.1f}  {U_all:.5f}  {err_blk:.5f}   {err_rep:.5f}   {err_delta:.5f}   {tau_u:6.2f}      | {Uc:.5f}  {Uh:.5f}  {z_hc:+.2f} | {Ua:.5f}  {Ub:.5f}  {z_half:+.2f}  | {R*nm}")

print()
print("Comparison with Metropolis:")
for k, v in metro.items():
    T = float(k); s = summary[T]
    z = (s["U"]-v["U"])/np.sqrt(s["err_blk"]**2+v["err"]**2)
    print(f"T={T}: Wolff {s['U']:.5f}+-{s['err_blk']:.5f}   Metropolis {v['U']:.5f}+-{v['err']:.5f}   z={z:+.2f}")

# thermalisation traces: ensemble mean of m^2 vs time for hot and cold starts
print()
print("Thermalisation (mean m^2 over replicas in windows of flips) -- cold vs hot, and equilibrium value")
for T in [1.5, 2.0, 2.2, 2.3, 2.4, 2.8]:
    d = np.load(f"/work/wolff_T{T}.npz"); th = d["th"].astype(float)/256; hot = d["hot"].astype(bool)
    eq = (d["M"].astype(float)/256)**2
    eqv = eq.mean()
    wins = [(0,5),(5,20),(20,50),(50,100),(100,200),(200,500),(500,1000),(1000,5000)]
    row=[]
    for (a_,b_) in wins:
        c = (th[~hot][:, a_:b_]**2).mean(); h_ = (th[hot][:, a_:b_]**2).mean()
        row.append(f"[{a_}-{b_}) c={c:.3f} h={h_:.3f}")
    print(f"T={T} eq<m2>={eqv:.4f} | " + " ; ".join(row))
json.dump({str(k):v for k,v in summary.items()}, open("/work/summary_v1.json","w"), indent=1)
