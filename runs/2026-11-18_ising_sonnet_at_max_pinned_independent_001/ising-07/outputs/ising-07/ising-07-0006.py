
code = r'''
import numpy as np, time, os
from sw import SWBatch

NB = 20  # number of time blocks per chain stored

def run_batch(N, T, R, neq, steps, seed, nb=NB, lagmax=60, record_burn=0):
    """Run R independent chains (half hot / half cold start).  Returns dict with
       blocks: (nb,R,4) block means of [m2_direct, m4_direct, m2_impr, m4_impr]
       acf_d, acf_i: chain-averaged normalized autocorrelation of m2 (direct / improved)
       burn: (record_burn,R,2) early-time m2 (direct, improved) for equilibration diagnostics"""
    Ns = N * N
    start = (np.arange(R) % 2)          # odd chains: cold, even chains: hot
    sim = SWBatch(N, R, T, seed=seed, start=start)
    burn = np.empty((record_burn, R, 2))
    for t in range(neq):
        M, S2, S4 = sim.step()
        if t < record_burn:
            burn[t, :, 0] = (M / Ns) ** 2
            burn[t, :, 1] = S2 / Ns ** 2
    L = steps // nb
    steps = L * nb
    x_d = np.empty((steps, R)); x_i = np.empty((steps, R))
    blocks = np.zeros((nb, R, 4))
    for b in range(nb):
        acc = np.zeros((R, 4))
        for t in range(L):
            M, S2, S4 = sim.step()
            m2 = (M / Ns) ** 2
            i2 = S2 / Ns ** 2
            acc[:, 0] += m2
            acc[:, 1] += m2 * m2
            acc[:, 2] += i2
            acc[:, 3] += (3 * S2 ** 2 - 2 * S4) / Ns ** 4
            x_d[b * L + t] = m2
            x_i[b * L + t] = i2
        blocks[b] = acc / L
    out = dict(blocks=blocks, N=N, T=T, R=R, neq=neq, steps=steps, seed=seed, burn=burn)
    for name, x in (("acf_d", x_d), ("acf_i", x_i)):
        out[name] = acf_chain_avg(x, lagmax)
    return out

def acf_chain_avg(x, lagmax):
    """x: (steps,R).  Chain-averaged autocovariance about the global mean, normalized."""
    steps, R = x.shape
    xc = x - x.mean()
    n = 1 << int(np.ceil(np.log2(2 * steps)))
    f = np.fft.rfft(xc, n=n, axis=0)
    ac = np.fft.irfft(f * np.conj(f), n=n, axis=0)[:lagmax + 1]
    ac = ac.sum(axis=1) / R
    ac = ac / (steps - np.arange(lagmax + 1))[:, None].ravel() if False else ac / (steps - np.arange(lagmax + 1))
    return ac / ac[0]

def tau_int(rho, c=6.0):
    """Sokal automatic windowing; returns (tau_int, window)"""
    tau = 0.5
    for k in range(1, len(rho)):
        tau += rho[k]
        if k >= c * tau:
            return tau, k
    return tau, len(rho) - 1

def Ufun(m2, m4):
    return 1.0 - m4 / (3.0 * m2 ** 2)

def jackknife_units(units):
    """units: (n,2) array of per-unit means of (m2,m4), equal-weight units.
       returns U (pooled), jackknife error."""
    n = units.shape[0]
    tot = units.sum(axis=0)
    U = Ufun(tot[0] / n, tot[1] / n)
    loo = (tot[None, :] - units) / (n - 1)
    Ui = Ufun(loo[:, 0], loo[:, 1])
    err = np.sqrt((n - 1) / n * ((Ui - Ui.mean()) ** 2).sum())
    # jackknife bias-corrected value too
    Ubc = n * U - (n - 1) * Ui.mean()
    return U, err, Ubc

def analyze(blocks, which):
    """blocks: (nb,R,4); which: 'd' or 'i'.  Returns dict of U, errors by several methods."""
    sel = [0, 1] if which == 'd' else [2, 3]
    b = blocks[:, :, sel]                     # (nb,R,2)
    nb, R, _ = b.shape
    chain = b.mean(axis=0)                    # (R,2) per-chain means
    U, e_chain, Ubc = jackknife_units(chain)
    U2, e_block, _ = jackknife_units(b.reshape(nb * R, 2))
    hot = chain[0::2]; cold = chain[1::2]
    Uh, eh, _ = jackknife_units(hot); Uc, ec, _ = jackknife_units(cold)
    return dict(U=U, err_chain=e_chain, err_block=e_block, U_biascorr=Ubc,
                U_hot=Uh, err_hot=eh, U_cold=Uc, err_cold=ec,
                m2=chain[:, 0].mean(), m4=chain[:, 1].mean())
'''
open('/work/prod.py','w').write(code)

import sys, time
sys.path.insert(0,'/work')
import numpy as np
import importlib, prod
importlib.reload(prod)
from prod import *

Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
N=16
t0=time.time()
print("  T    | U_direct +- err(chain)/err(block) | U_impr +- err(chain)/err(block) | tau_d tau_i | hot-cold diff (impr) sigma | time")
for i,T in enumerate(Ts):
    out = run_batch(N,T,R=200,neq=300,steps=1000,seed=5000+i,record_burn=0)
    ad = analyze(out['blocks'],'d'); ai = analyze(out['blocks'],'i')
    td,_ = tau_int(out['acf_d']); ti,_ = tau_int(out['acf_i'])
    dd = (ai['U_hot']-ai['U_cold'])/np.hypot(ai['err_hot'],ai['err_cold'])
    print("%.2f | %.4f +- %.4f / %.4f | %.4f +- %.4f / %.4f | %.2f %.2f | %.2f | %.0fs"%(T,ad['U'],ad['err_chain'],ad['err_block'],ai['U'],ai['err_chain'],ai['err_block'],td,ti,dd,time.time()-t0))
