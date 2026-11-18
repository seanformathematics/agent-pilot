import sys, json, math
sys.path.insert(0, '/work')
import numpy as np
from wolff_lib import U_of, jackknife

Ts = [round(1.5 + 0.1*i, 1) for i in range(14)]
K = 8
prefixes = ['T', 'B2_T', 'B3_T', 'B4_T']
def load_batch(T, p):
    return [np.load(f'/work/data/{p}{T:.1f}_c{k}.npy').astype(np.int64) for k in range(K)]
def blocks_from(chains, bsize):
    out = []
    for c in chains:
        nb = len(c)//bsize
        out += [c[i*bsize:(i+1)*bsize] for i in range(nb)]
    return out

rng = np.random.default_rng(2024)
final = {}
chi2_sum = 0.0
print(f"{'T':>4} {'U':>9} {'U_err':>8} {'jk_blk5k':>9} {'jk_chain':>9} {'scat/sqrt32':>11} {'boot_chain':>10} {'chi2_b(3)':>9} {'n_meas':>9}")
for T in Ts:
    batches = [load_batch(T, p) for p in prefixes]
    chains = sum(batches, [])
    nch = len(chains)
    Uall, e_blk = jackknife(blocks_from(chains, 5000))
    _, e_chain = jackknife(chains)
    Uc = np.array([U_of(c) for c in chains])
    e_scat = Uc.std(ddof=1)/math.sqrt(nch)
    # chain-level bootstrap of the pooled estimator
    bs = []
    for b in range(2000):
        idx = rng.integers(0, nch, nch)
        sx = sum(((chains[i].astype(np.float64))**2).sum() for i in idx)
        sy = sum((((chains[i].astype(np.float64))**2)**2).sum() for i in idx)
        nn = sum(len(chains[i]) for i in idx)
        bs.append(1.0 - (sy/nn)/(3.0*(sx/nn)**2))
    e_boot = float(np.std(bs, ddof=1))
    Ub = np.array([jackknife(blocks_from(b, 5000))[0] for b in batches])
    eb = np.array([jackknife(blocks_from(b, 5000))[1] for b in batches])
    w = 1/eb**2
    Uw = np.sum(w*Ub)/np.sum(w)
    chi2 = float(np.sum((Ub-Uw)**2*w))
    chi2_sum += chi2
    nmeas = int(sum(len(c) for c in chains))
    final[T] = dict(U=float(Uall), e_blk=e_blk, e_chain=e_chain, e_scat=e_scat, e_boot=e_boot, chi2=chi2, n=nmeas, nch=nch)
    print(f"{T:4.1f} {Uall:9.5f} {max(e_blk,e_scat):8.5f} {e_blk:9.5f} {e_chain:9.5f} {e_scat:11.5f} {e_boot:10.5f} {chi2:9.2f} {nmeas:9d}")
print("sum of batch chi2 over 14 T (4 batches, 3 dof each; expect 42 +/- 9.2):", round(chi2_sum,2))
json.dump({str(k): v for k, v in final.items()}, open('/work/final_v4.json','w'), indent=1)
