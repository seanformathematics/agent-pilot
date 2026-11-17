
import os, sys, time, math, json
import numpy as np
os.chdir('/work'); sys.path.insert(0, '/work')
import wolff_py as W

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
NCH = 4
STEPS_PER_CHAIN = 500000
BL = 100
NTHERM = 10000
BUDGET = 460.0
nbrs = W.make_nbrs(16)

def fname(T):
    return f'/work/results/T{T:.2f}.npz'

t0 = time.time()
for i, T in enumerate(Ts):
    if os.path.exists(fname(T)):
        continue
    if time.time() - t0 > BUDGET:
        print('budget reached before T=', T, flush=True)
        break
    t1 = time.time()
    S2s, S4s, seeds = [], [], []
    nfl = 0
    for c in range(NCH):
        seed = 1000*(i+1) + c + 1
        seeds.append(seed)
        S2, S4, nf = W.wolff_run(16, T, seed, NTHERM, STEPS_PER_CHAIN//BL, BL, nbrs)
        S2s.append(S2); S4s.append(S4); nfl += nf
    S2 = np.concatenate(S2s); S4 = np.concatenate(S4s)
    np.savez(fname(T), S2=S2, S4=S4, seeds=np.array(seeds), nflip=nfl)
    U, err = W.jk_U(S2, S4, BL, J=100)
    print(f'T={T}: U={U:.5f} jk_err={err:.5f}  time={time.time()-t1:.1f}s', flush=True)

print('production elapsed', round(time.time()-t0, 1), flush=True)

# ---------- analysis of all completed temperatures ----------
rows = []
for i, T in enumerate(Ts):
    if not os.path.exists(fname(T)):
        continue
    d = np.load(fname(T))
    S2 = d['S2'].astype(np.int64); S4 = d['S4'].astype(np.int64)
    nch = NCH; nbl = STEPS_PER_CHAIN//BL
    n_tot = float(len(S2)*BL)
    U_full = 1 - (S4.sum()/n_tot)/(3*(S2.sum()/n_tot)**2)
    errs = {J: W.jk_U(S2, S4, BL, J=J)[1] for J in [20, 50, 100, 200]}
    Uc = []
    for c in range(nch):
        a = S2[c*nbl:(c+1)*nbl]; b = S4[c*nbl:(c+1)*nbl]
        Uc.append(1 - (b.sum()/(nbl*BL))/(3*(a.sum()/(nbl*BL))**2))
    Uc = np.array(Uc)
    chain_err = Uc.std(ddof=1)/np.sqrt(nch)
    # lag-1 autocorrelation of block sums (independence check)
    x = S2.astype(float); x = x - x.mean()
    rho1 = np.sum(x[:-1]*x[1:])/np.sum(x*x)
    print(f'T={T:.2f}: U={U_full:.5f}  jk_err(J=20,50,100,200)=' + ', '.join(f'{errs[J]:.5f}' for J in [20,50,100,200])
          + f'  chain_std_err={chain_err:.5f}  chain U=' + ' '.join(f'{u:.4f}' for u in Uc)
          + f'  rho1_blocks={rho1:.3f}  flips={int(d["nflip"])}', flush=True)
    rows.append(dict(T=T, U=float(U_full), err=float(errs[100]), err_J20=float(errs[20]), err_J200=float(errs[200]), chain_err=float(chain_err), n=int(n_tot)))
json.dump(rows, open('/work/results/summary_partial.json', 'w'), indent=1)
