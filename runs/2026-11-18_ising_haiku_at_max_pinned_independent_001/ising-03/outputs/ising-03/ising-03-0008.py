import sys, json, math
sys.path.insert(0, '/work')
import numpy as np
from wolff_lib import U_of, jackknife

Ts = [round(1.5 + 0.1*i, 1) for i in range(14)]
K = 8
def load(T, batch):
    if batch == 1:
        return [np.load(f'/work/data/T{T:.1f}_c{k}.npy').astype(np.int64) for k in range(K)]
    return [np.load(f'/work/data/B2_T{T:.1f}_c{k}.npy').astype(np.int64) for k in range(K)]

def blocks_from(chains, bsize):
    out = []
    for c in chains:
        nb = len(c)//bsize
        out += [c[i*bsize:(i+1)*bsize] for i in range(nb)]
    return out

final = {}
print(f"{'T':>4} {'U_all':>9} {'err_jk5k':>9} {'err_jk2.5k':>10} {'err_jk12.5k':>11} {'err_chain16':>11} {'U_batch1':>9} {'U_batch2':>9} {'dU':>8} {'z':>6}")
zs = []
for T in Ts:
    c1 = load(T, 1); c2 = load(T, 2)
    call = c1 + c2
    Uall, e5 = jackknife(blocks_from(call, 5000))
    _, e25 = jackknife(blocks_from(call, 2500))
    _, e125 = jackknife(blocks_from(call, 12500))
    _, echain = jackknife(call)
    U1, e1 = jackknife(blocks_from(c1, 5000))
    U2, e2 = jackknife(blocks_from(c2, 5000))
    dU = U1 - U2
    z = dU/math.sqrt(e1**2 + e2**2)
    zs.append(z)
    Uc = np.array([U_of(c) for c in call])
    scat = Uc.std(ddof=1)/math.sqrt(len(call))
    final[T] = dict(U=Uall, err=e5, n_updates=int(sum(len(c) for c in call)), err_chain16=echain, err_scatter=scat)
    print(f"{T:4.1f} {Uall:9.5f} {e5:9.5f} {e25:10.5f} {e125:11.5f} {echain:11.5f} {U1:9.5f} {U2:9.5f} {dU:8.5f} {z:6.2f}")
zs = np.array(zs)
print("z-scores of batch1-batch2 differences: mean=%.2f  std=%.2f (expect mean~0, std~1)" % (zs.mean(), zs.std(ddof=1)))
json.dump({str(k): v for k, v in final.items()}, open('/work/final_v2.json', 'w'), indent=1)
