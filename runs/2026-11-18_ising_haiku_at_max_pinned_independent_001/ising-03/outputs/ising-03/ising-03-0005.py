import sys, json, math
sys.path.insert(0, '/work')
import numpy as np
from wolff_lib import U_of, jackknife

Ts = [round(1.5 + 0.1*i, 1) for i in range(14)]
K = 8

def tau_int(x, c=6.0):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2*n)
    acf = np.fft.irfft(f*np.conj(f))[:n]
    acf = acf/acf[0]
    tau = 0.5
    for t in range(1, n):
        tau += acf[t]
        if t >= c*tau:
            return tau
    return tau

rows = []
summary = {}
for T in Ts:
    chains = [np.load(f'/work/data/T{T:.1f}_c{k}.npy').astype(np.int64) for k in range(K)]
    allM = np.concatenate(chains)
    Ufull = U_of(allM)
    # jackknife with blocks of 5000 updates (25 per chain)
    blocks5k = [c[i*5000:(i+1)*5000] for c in chains for i in range(25)]
    _, e5k = jackknife(blocks5k)
    blocks2500 = [c[i*2500:(i+1)*2500] for c in chains for i in range(50)]
    _, e2500 = jackknife(blocks2500)
    blocks_chain = [c for c in chains]
    _, echain = jackknife(blocks_chain)
    # chain-to-chain scatter
    Uc = np.array([U_of(c) for c in chains])
    scat = Uc.std(ddof=1)/math.sqrt(K)
    tx = tau_int((allM[:125000].astype(float))**2) if False else tau_int((chains[0].astype(float))**2)
    ty = tau_int((chains[0].astype(float))**4)
    absm = np.mean(np.abs(allM))/256
    m2 = np.mean((allM/256.0)**2)
    summary[T] = dict(U=Ufull, err_blk5000=e5k, err_blk2500=e2500, err_chain=echain, err_scatter=scat,
                      chainU_std=Uc.std(ddof=1), tau_m2=tx, tau_m4=ty, absm=absm, m2=m2)
    print(f"T={T:.1f}  U={Ufull:.5f}  err(jk,5k blk)={e5k:.5f}  err(jk,2.5k blk)={e2500:.5f}  err(jk,chain)={echain:.5f}  "
          f"err(scatter/sqrt8)={scat:.5f}  tau(M2)={tx:.2f} tau(M4)={ty:.2f}  <|m|>={absm:.4f} <m2>={m2:.4f}")
json.dump({str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in summary.items()}, open('/work/summary_v1.json','w'), indent=1)
