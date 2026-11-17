import os, time, math, random, sys, json
sys.path.insert(0, '/work')
import numpy as np
import ising_wolff as iw

def batch(T, L, seeds, nflip, blk, discard):
    N = L*L; nb = iw.make_nb(L); padd = 1 - math.exp(-2.0/T)
    S2_all = []; S4_all = []
    for seed in seeds:
        rnd = random.Random(seed).random
        spins = [1 if rnd() < 0.5 else -1 for _ in range(N)]
        mark = [0]*N
        M = sum(spins)
        M, stamp, _, _ = iw.run_flips(spins, mark, 0, M, discard, padd, rnd, nb, N)
        nblk = nflip//blk
        S2b = np.empty(nblk); S4b = np.empty(nblk)
        for b in range(nblk):
            M, stamp, s2, s4 = iw.run_flips(spins, mark, stamp, M, blk, padd, rnd, nb, N)
            S2b[b] = s2; S4b[b] = s4
        assert sum(spins) == M
        S2_all.append(S2b); S4_all.append(S4b)
    return np.array(S2_all), np.array(S4_all)

t_start = time.time()
for j, T in enumerate([1.5, 1.6, 1.7]):
    fn = f'/work/res3_{T:.2f}.npz'
    if os.path.exists(fn):
        continue
    t0 = time.time()
    seeds = [70000 + 10*j + k for k in range(4)]
    S2a, S4a = batch(T, 16, seeds, 300000, 1000, 2000)
    np.savez(fn, S2b=S2a, S4b=S4a, seeds=np.array(seeds), T=T, nflip=300000, blk=1000, discard=2000)
    U, e = iw.jk_U(S2a.ravel(), S4a.ravel(), np.full(S2a.size, 1000.0))
    print(f'new batch T={T:.2f} U={U:.7f} jk_err={e:.8f} time={time.time()-t0:.1f}s', flush=True)
print('elapsed', round(time.time()-t_start, 1), flush=True)

# ---- compile final results ----
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
final = []
for T in Ts:
    files = [f'/work/res_{T:.2f}.npz', f'/work/res2_{T:.2f}.npz', f'/work/res3_{T:.2f}.npz']
    files = [f for f in files if os.path.exists(f)]
    S2s = []; S4s = []; seeds = []
    batch_info = []
    for f in files:
        d = np.load(f)
        assert int(d['blk']) == 1000
        S2s.append(d['S2b']); S4s.append(d['S4b']); seeds += [int(x) for x in d['seeds']]
        Ub, eb = iw.jk_U(d['S2b'].ravel(), d['S4b'].ravel(), np.full(d['S2b'].size, 1000.0))
        batch_info.append((os.path.basename(f), round(Ub, 6), round(eb, 6), int(d['nflip'])*d['S2b'].shape[0]))
    S2 = np.concatenate([a.ravel() for a in S2s]); S4 = np.concatenate([a.ravel() for a in S4s])
    cnt = np.full(S2.size, 1000.0)
    U, e = iw.jk_U(S2, S4, cnt)
    n_samp = int(cnt.sum())
    final.append(dict(T=T, U=U, U_err=e, n_samples=n_samp, seeds=seeds, batches=batch_info))
    print(f'T={T:.2f}  U={U:.6f}  U_err={e:.6f}  n_flips={n_samp}  seeds={seeds}')
    for bi in batch_info:
        print('      ', bi)

with open('/work/final_results.json', 'w') as fh:
    json.dump(final, fh, indent=1)
print('max U_err =', max(r['U_err'] for r in final))
print('all finite and <= 2/3:', all(np.isfinite(r['U']) and r['U'] <= 2/3 for r in final))
