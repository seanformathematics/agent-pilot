
import sys, time, math, json
sys.path.insert(0, "/work")
import numpy as np
import ising

N = 16; NN = 256
def U_of(M):
    m = np.asarray(M, dtype=np.float64)/NN
    return 1.0 - np.mean(m**4)/(3*np.mean(m**2)**2)

def jk_single(M, bpr=100):
    m = np.asarray(M, dtype=np.float64)/NN
    m2 = m*m; m4 = m2*m2
    L = len(m)//bpr
    S2 = m2[:L*bpr].reshape(bpr, L).sum(axis=1); S4 = m4[:L*bpr].reshape(bpr, L).sum(axis=1)
    T2 = S2.sum(); T4 = S4.sum(); n = bpr*L
    s2 = (T2-S2)/((bpr-1)*L); s4 = (T4-S4)/((bpr-1)*L)
    Uj = 1 - s4/(3*s2*s2)
    return math.sqrt((bpr-1)/bpr*np.sum((Uj-Uj.mean())**2))

t0 = time.time()
summary = {}
for ti, T in enumerate([2.6, 2.7, 2.8]):
    Us = []; jks = []
    for r in range(16):
        sd = 9000000 + 1000*ti + r
        M, _ = ising.simulate(N, T, 5000, 200000, seed=sd, start="hot")
        Us.append(U_of(M)); jks.append(jk_single(M))
    Us = np.array(Us)
    summary[T] = dict(mean=Us.mean(), sd=Us.std(ddof=1), se_of_mean=Us.std(ddof=1)/4.0, jk_single_mean=float(np.mean(jks)))
    print(f"T={T}: 16 fresh replicas x 200k: mean U={Us.mean():.5f}, sd across replicas={Us.std(ddof=1):.5f}, "
          f"mean single-replica jackknife err={np.mean(jks):.5f}, se of 16-rep mean={Us.std(ddof=1)/4:.5f}; "
          f"time so far {time.time()-t0:.1f}s", flush=True)

# burn-in / drift check on saved production and pilot series at T=2.8
prod = np.load("/work/prod_T2.8.npy").astype(np.int64)
pilot = np.load("/work/pilot_T2.8.npy").astype(np.int64)
for name, arr in [("prod rep0", prod[0]), ("prod rep1", prod[1]), ("prod rep2", prod[2]), ("prod rep3", prod[3]), ("pilot", pilot)]:
    chunks = arr[:len(arr)//8*8].reshape(8, -1)
    print(name, "U per 1/8 chunk:", np.round([U_of(c) for c in chunks], 4).tolist())
print("first 2000 vs rest, pilot:", round(U_of(pilot[:2000]),4), round(U_of(pilot[2000:]),4))
