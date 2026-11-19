
import sys, json
sys.path.insert(0,"/work")
import numpy as np
from analysis import *

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
metro = json.load(open("/work/metropolis_validation.json"))

def load_pooled(T):
    ms2=[]; ms4=[]; cb=[]
    for prefix in ("wolff","wolffB"):
        d = np.load(f"/work/{prefix}_T{T}.npz")
        M = d["M"]; R, nm = M.shape
        m2, m4 = moments(M.reshape(-1))
        ms2.append(m2.reshape(R, nm)); ms4.append(m4.reshape(R, nm)); cb.append(d["cbar"])
    return np.concatenate(ms2), np.concatenate(ms4), np.concatenate(cb)

def blockjk(m2, m4, B):
    R, nm = m2.shape
    L = nm//B
    S2b = m2[:, :L*B].reshape(R, B, L).sum(2).reshape(-1); S4b = m4[:, :L*B].reshape(R, B, L).sum(2).reshape(-1)
    return jackknife_U(S2b, S4b, np.full(R*B, L))

final = {}
print(" T     U            err_blk(B=20)  | B=1(rep) B=2     B=5     B=10    B=50    B=100  | err_delta  | tau_u  tau_m2  acf_m2(lag100) | n_total   cbar   Ubc-U")
for T in Ts:
    m2, m4, cb = load_pooled(T)
    R, nm = m2.shape
    U20, e20, Ubc20 = blockjk(m2, m4, 20)
    es = {B: blockjk(m2, m4, B)[1] for B in (1, 2, 5, 10, 50, 100)}
    # delta method with Sokal windows per replica
    a = m2.mean(); b = m4.mean()
    taus=[]; var_means=[]; tau2s=[]; acf100=[]
    for r in range(R):
        u = -(m4[r]-b)/(3*a*a) + (2*b/(3*a**3))*(m2[r]-a)
        tau, W = tau_int_sokal(u); taus.append(tau); var_means.append(u.var()*tau/nm)
        x = m2[r]-m2[r].mean(); f = np.fft.rfft(x, 2*nm); acf = np.fft.irfft(f*np.conj(f))[:nm]; acf/=acf[0]
        acf100.append(acf[100]); tau2s.append(tau_int_sokal(m2[r])[0])
    err_delta = np.sqrt(np.sum(var_means))/R
    final[T] = dict(U=float(U20), err_blk20=float(e20), err_rep=float(es[1]), err_delta=float(err_delta),
                    n=int(R*nm), tau_u=float(np.mean(taus)), tau_m2=float(np.mean(tau2s)), cbar=float(cb.mean()),
                    Ubc=float(Ubc20), err_by_B={str(k): float(v) for k, v in es.items()})
    print(f"{T:.1f}  {U20:.7f}  {e20:.2e}    | {es[1]:.2e} {es[2]:.2e} {es[5]:.2e} {es[10]:.2e} {es[50]:.2e} {es[100]:.2e} | {err_delta:.2e}  | {np.mean(taus):5.2f}  {np.mean(tau2s):5.2f}  {np.mean(acf100):+.4f}  | {R*nm}  {cb.mean():6.1f}  {Ubc20-U20:+.1e}")

print("\nWolff(A+B) vs Metropolis:")
zs=[]
for k in sorted(metro, key=float):
    v = metro[k]; T=float(k); s=final[T]
    z = (s["U"]-v["U"])/np.sqrt(s["err_blk20"]**2+v["err"]**2); zs.append(z)
    print(f"T={T}: Wolff {s['U']:.6f} +- {s['err_blk20']:.2e} | Metropolis {v['U']:.6f} +- {v['err']:.2e} | diff={s['U']-v['U']:+.2e} z={z:+.2f}")
print("sum z^2 =", np.sum(np.square(zs)), "for", len(zs), "points")
json.dump({str(k): v for k, v in final.items()}, open("/work/final_results_v1.json", "w"), indent=1)
