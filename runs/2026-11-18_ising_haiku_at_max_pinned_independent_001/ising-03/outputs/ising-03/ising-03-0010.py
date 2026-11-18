import sys, json, math
sys.path.insert(0, '/work')
import numpy as np
from wolff_lib import U_of, jackknife

Ts = [round(1.5 + 0.1*i, 1) for i in range(14)]
K = 8
prefix = {1: 'T', 2: 'B2_T', 3: 'B3_T'}
def load(T, batch):
    p = prefix[batch]
    return [np.load(f'/work/data/{p}{T:.1f}_c{k}.npy').astype(np.int64) for k in range(K)]
def blocks_from(chains, bsize):
    out = []
    for c in chains:
        nb = len(c)//bsize
        out += [c[i*bsize:(i+1)*bsize] for i in range(nb)]
    return out

results = {}
chi2_total = 0.0
print(f"{'T':>4} {'U':>9} {'err_jk5k':>10} {'err_jk1250':>11} {'err_jk25k':>10} {'err_chain24':>11} {'U_b1':>8} {'U_b2':>8} {'U_b3':>8} {'chi2(2dof)':>10}")
for T in Ts:
    cb = {b: load(T, b) for b in (1, 2, 3)}
    call = cb[1] + cb[2] + cb[3]
    Uall, e5 = jackknife(blocks_from(call, 5000))
    _, e1250 = jackknife(blocks_from(call, 1250))
    _, e25k = jackknife(blocks_from(call, 25000))
    _, echain = jackknife(call)
    Ub, eb = [], []
    for b in (1, 2, 3):
        u, e = jackknife(blocks_from(cb[b], 5000))
        Ub.append(u); eb.append(e)
    Ub = np.array(Ub); eb = np.array(eb)
    # chi-square of batch means about the pooled mean (approx, 2 dof)
    chi2 = float(np.sum((Ub - Uall)**2/eb**2))
    chi2_total += chi2
    Uc = np.array([U_of(c) for c in call])
    scat = Uc.std(ddof=1)/math.sqrt(len(call))
    results[T] = dict(U=Uall, err=e5, n_updates=int(sum(len(c) for c in call)), err_jk1250=e1250, err_jk25k=e25k,
                      err_chain=echain, err_scatter=scat, chi2=chi2)
    print(f"{T:4.1f} {Uall:9.5f} {e5:10.5f} {e1250:11.5f} {e25k:10.5f} {echain:11.5f} {Ub[0]:8.5f} {Ub[1]:8.5f} {Ub[2]:8.5f} {chi2:10.2f}")
print("sum chi2 over 14 T (expect ~28 +/- 7.5 if errors are calibrated):", round(chi2_total,2))
json.dump({str(k): v for k, v in results.items()}, open('/work/final_v3.json','w'), indent=1)
