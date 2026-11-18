import sys, json, math
sys.path.insert(0, '/work')
import numpy as np
from wolff_lib import U_of, jackknife

Ts = [round(1.5 + 0.1*i, 1) for i in range(14)]
K = 8
bases = {'T': 0, 'B2_T': 100000, 'B3_T': 200000, 'B4_T': 300000}
prefixes = ['T', 'B2_T', 'B3_T', 'B4_T']
def load_batch(T, p):
    return [np.load(f'/work/data/{p}{T:.1f}_c{k}.npy').astype(np.int64) for k in range(K)]
def blocks_from(chains, bsize):
    out = []
    for c in chains:
        nb = len(c)//bsize
        out += [c[i*bsize:(i+1)*bsize] for i in range(nb)]
    return out

rows = {}
chi2_sum = 0.0
for T in Ts:
    batches = [load_batch(T, p) for p in prefixes]
    chains = sum(batches, [])
    Uall, e_blk = jackknife(blocks_from(chains, 5000))
    Uc = np.array([U_of(c) for c in chains])
    e_scat = Uc.std(ddof=1)/math.sqrt(len(chains))
    Ub = np.array([jackknife(blocks_from(b, 5000))[0] for b in batches])
    eb = np.array([jackknife(blocks_from(b, 5000))[1] for b in batches])
    w = 1/eb**2; Uw = np.sum(w*Ub)/np.sum(w)
    chi2 = float(np.sum((Ub-Uw)**2*w)); chi2_sum += chi2
    rows[T] = dict(U=float(Uall), e_blk=float(e_blk), e_scat=float(e_scat), chi2=chi2,
                   n=int(sum(len(c) for c in chains)))
factor = math.sqrt(max(1.0, chi2_sum/42.0))
print("calibration factor sqrt(chi2_sum/42) =", round(math.sqrt(chi2_sum/42.0),4), " applied factor =", round(factor,4))
print(f"{'T':>4} {'U':>12} {'U_err':>12} {'e_blk':>12} {'e_scat':>12} {'n':>9}")
for T in Ts:
    r = rows[T]
    uerr = max(r['e_blk'], r['e_scat'])*factor
    r['U_err_final'] = uerr
    print(f"{T:4.1f} {r['U']:12.7f} {uerr:12.7f} {r['e_blk']:12.7f} {r['e_scat']:12.7f} {r['n']:9d}")
json.dump({str(T): rows[T] for T in Ts}, open('/work/final_table.json','w'), indent=1)

# seed list
seeds = []
for iT, T in enumerate(Ts):
    for bi, p in enumerate(prefixes):
        for k in range(K):
            seeds.append(bases[p] + 1000*(iT+1) + k + 17)
print("number of seeds:", len(seeds), " unique:", len(set(seeds)))
print(json.dumps(seeds))
