
import sys, time, numpy as np
sys.path.insert(0, '/work')
import sw

def metro_run(L, R, T, seed, burn, S):
    rng = np.random.default_rng(seed)
    s = (2*rng.integers(0, 2, size=(R, L, L)) - 1).astype(np.int8)
    x = np.arange(L)[:, None]; y = np.arange(L)[None, :]
    par = (x + y) % 2
    masks = [(par == 0)[None, :, :], (par == 1)[None, :, :]]
    acc = np.ones(17)
    for dE in [4, 8]:
        acc[dE+8] = np.exp(-dE/T)
    def half(s, mask):
        nb = np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2)
        dE = (2*s*nb).astype(np.int64)
        p = acc[dE + 8]
        u = rng.random(s.shape)
        flip = mask & (u < p)
        return np.where(flip, -s, s).astype(np.int8)
    B2 = np.zeros(R); B4 = np.zeros(R); NN = np.zeros(R); ncnn = 0
    for t in range(burn + S):
        s = half(s, masks[0]); s = half(s, masks[1])
        if t >= burn:
            M = s.sum(axis=(1, 2), dtype=np.int64)
            m = M/(L*L); m2 = m*m
            B2 += m2; B4 += m2*m2
            if (t - burn) % 5 == 0:
                NN += (s*np.roll(s, -1, 1) + s*np.roll(s, -1, 2)).sum(axis=(1, 2), dtype=np.int64)/(2*L*L)
                ncnn += 1
    return B2, B4, NN/ncnn

t0 = time.time()
for T in [2.3, 2.5, 2.8]:
    seed = 70000 + int(round(T*10))
    B2, B4, NN = metro_run(16, 1000, T, seed, burn=300, S=1000)
    U, err = sw.binder_from_sums(B2, B4, 1000)
    # compare with SW pooled results
    d1 = np.load(f'/work/res_T{T:.2f}.npz'); d2 = np.load(f'/work/rep2_T{T:.2f}.npz')
    nn_sw = np.concatenate([d1['NN'], d2['NN']]).mean()
    print(f"METRO T={T:.2f} U={U:.5f} +- {err:.5f} | SW pooled nn={nn_sw:.5f}, Metro nn={NN.mean():.5f} +- {NN.std(ddof=1)/np.sqrt(len(NN)):.5f} | elapsed={time.time()-t0:.1f}", flush=True)
pooled = np.load('/work/pooled.npy')
for row in pooled:
    if abs(row[0]-2.3)<1e-9 or abs(row[0]-2.5)<1e-9 or abs(row[0]-2.8)<1e-9:
        print("SW pooled U at T=%.2f: %.6f +- %.6f" % (row[0], row[1], row[2]))
