
import sys, json
sys.path.insert(0,"/work")
import numpy as np
from analysis import *

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]

def load(prefix, T):
    d = np.load(f"/work/{prefix}_T{T}.npz")
    M = d["M"]; hot = d["hot"].astype(bool)
    m2, m4 = moments(M.reshape(-1))
    R, nm = M.shape
    return m2.reshape(R, nm), m4.reshape(R, nm), hot

def est(m2, m4, B=None):
    """pooled U with block-jackknife error (R replicas x B blocks) and replica-level jackknife error"""
    R, nm = m2.shape
    S2r = m2.sum(1); S4r = m4.sum(1); cr = np.full(R, nm)
    U, e_rep, _ = jackknife_U(S2r, S4r, cr)
    if B is None: B = 10 if nm >= 20000 else 5
    L = nm//B
    S2b = m2[:, :L*B].reshape(R, B, L).sum(2).reshape(-1); S4b = m4[:, :L*B].reshape(R, B, L).sum(2).reshape(-1)
    U2, e_blk, Ubc = jackknife_U(S2b, S4b, np.full(R*B, L))
    return U, e_blk, e_rep, Ubc

zs_AB=[]; zs_hcB=[]; zs_hcAll=[]
print("T    U_A         err_A      U_B         err_B      z_AB  | hot-cold in B: z_hcB | pooled(A+B): hot-cold z, ")
for T in Ts:
    m2A, m4A, hotA = load("wolff", T); m2B, m4B, hotB = load("wolffB", T)
    UA, eA, erA, _ = est(m2A, m4A); UB, eB, erB, _ = est(m2B, m4B)
    zAB = (UA-UB)/np.sqrt(eA**2+eB**2)
    # hot vs cold in B
    Uc, ec, _, _ = est(m2B[~hotB], m4B[~hotB]); Uh, eh, _, _ = est(m2B[hotB], m4B[hotB])
    zhB = (Uc-Uh)/np.sqrt(ec**2+eh**2)
    # pooled hot vs cold
    m2all = np.concatenate([m2A, m2B]); m4all = np.concatenate([m4A, m4B]); hall = np.concatenate([hotA, hotB])
    Uc2, ec2, _, _ = est(m2all[~hall], m4all[~hall]); Uh2, eh2, _, _ = est(m2all[hall], m4all[hall])
    zh2 = (Uc2-Uh2)/np.sqrt(ec2**2+eh2**2)
    zs_AB.append(zAB); zs_hcB.append(zhB); zs_hcAll.append(zh2)
    print(f"{T:.1f}  {UA:.6f}  {eA:.6f}   {UB:.6f}  {eB:.6f}   {zAB:+.2f}  |  {zhB:+.2f}  |  {zh2:+.2f}   (dU hot-cold pooled = {Uc2-Uh2:+.2e})")
zs_AB=np.array(zs_AB); zs_hcB=np.array(zs_hcB); zs_hcAll=np.array(zs_hcAll)
print("\nsum z^2 (A vs B) =", np.sum(zs_AB**2), " for 14 dof;   rms z =", np.sqrt(np.mean(zs_AB**2)))
print("sum z^2 (hot-cold, B) =", np.sum(zs_hcB**2), "; sum z^2 (hot-cold, pooled) =", np.sum(zs_hcAll**2))
