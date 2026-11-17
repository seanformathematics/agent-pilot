import numpy as np, time, sys
src = r'''
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def sw_steps_nd(S, p, rng, N):
    """One Swendsen-Wang step for R replicas. S: (R,N,N) int8 array (modified copy returned)."""
    R = S.shape[0]
    W = 2 * N + 1
    bh = (S == np.roll(S, -1, axis=2)) & (rng.random((R, N, N)) < p)
    bv = (S == np.roll(S, -1, axis=1)) & (rng.random((R, N, N)) < p)
    I = np.zeros((R, W, W), dtype=bool)
    ev = np.arange(0, 2 * N, 2)
    I[:, 0:2 * N:2, 0:2 * N:2] = True          # node cells
    I[:, 2 * N, 0:2 * N:2] = True              # copy of row 0 nodes (periodic image, vertical wrap)
    I[:, 0:2 * N:2, 2 * N] = True              # copy of column 0 nodes (horizontal wrap)
    I[:, 0:2 * N:2, 1:2 * N:2] = bh            # horizontal bonds (i,j)-(i,j+1)
    I[:, 1:2 * N:2, 0:2 * N:2] = bv            # vertical bonds (i,j)-(i+1,j)
    img = np.zeros((W, R, W + 1), dtype=bool)
    img[:, :, :W] = I.transpose(1, 0, 2)
    img2 = img.reshape(W, R * (W + 1))
    lab2, K = ndimage.label(img2)
    off = np.arange(R) * (W + 1)
    cols = off[:, None] + ev[None, :]                          # (R,N) node columns
    node_labels = lab2[ev[None, :, None], cols[:, None, :]]   # (R,N,N): node (i,j) of replica r
    # merge edges from periodic copies
    a1 = lab2[0, cols]; b1 = lab2[2 * N, cols]
    a2 = lab2[ev[None, :], off[:, None]]; b2 = lab2[ev[None, :], (off + 2 * N)[:, None]]
    src_ = np.concatenate([a1.ravel(), a2.ravel()])
    dst_ = np.concatenate([b1.ravel(), b2.ravel()])
    G = coo_matrix((np.ones(src_.size, dtype=np.int8), (src_, dst_)), shape=(K + 1, K + 1))
    ncomp, comp = connected_components(G, directed=False)
    site_comp = comp[node_labels]
    flags = rng.random(ncomp) < 0.5
    flip = flags[site_comp]
    return np.where(flip, -S, S).astype(np.int8)

def sw_run_nd(T, R, n_steps, n_burn, seed, N=16, S0=None):
    rng = np.random.default_rng(seed)
    p = 1.0 - np.exp(-2.0 / T)
    if S0 is None:
        S = (rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * np.int8(2) - np.int8(1)).astype(np.int8)
    else:
        S = S0.astype(np.int8).copy()
    M = np.empty((n_steps, R), dtype=np.int16)
    for step in range(n_burn + n_steps):
        S = sw_steps_nd(S, p, rng, N)
        if step >= n_burn:
            M[step - n_burn] = S.sum(axis=(1, 2), dtype=np.int32)
    return M, S
'''
open('/work/swnd.py', 'w').write(src)
sys.path.insert(0, '/work')
import swnd, sw
exec(open('/work/sw.py').read().split("'''")[0]) if False else None

def exact_U(N, T):
    NN = N * N
    nconf = 1 << NN
    configs = np.arange(nconf, dtype=np.int64)
    bits = (configs[:, None] >> np.arange(NN)) & 1
    S = (2 * bits - 1).astype(np.int32)
    idx = np.arange(NN).reshape(N, N)
    E = np.zeros(nconf)
    for i in range(N):
        for j in range(N):
            a = idx[i, j]
            E -= S[:, a] * S[:, idx[i, (j + 1) % N]]
            E -= S[:, a] * S[:, idx[(i + 1) % N, j]]
    M = S.sum(axis=1)
    m = M / NN
    logw = -E / T
    logw -= logw.max()
    w = np.exp(logw)
    Z = w.sum()
    m2 = (w * m ** 2).sum() / Z
    m4 = (w * m ** 4).sum() / Z
    return 1 - m4 / (3 * m2 ** 2), m2

# correctness on 4x4 vs exact
for T in [2.27, 3.0]:
    t0 = time.time()
    Ue, m2e = exact_U(4, T)
    M, _ = swnd.sw_run_nd(T, R=1000, n_steps=1000, n_burn=100, seed=99, N=4)
    m = M / 16.0
    Uf = 1 - (m**4).mean() / (3 * (m**2).mean()**2)
    print(f"[nd] N=4 T={T}: exact U={Ue:.5f}, nd U={Uf:.5f}, exact <m2>={m2e:.5f}, nd <m2>={(m**2).mean():.5f}, t={time.time()-t0:.1f}s")

# timing on 16x16
for R in [512]:
    t0 = time.time()
    M, S = swnd.sw_run_nd(2.2692, R=R, n_steps=50, n_burn=10, seed=7, N=16)
    dt = time.time() - t0
    print(f"[nd] N=16 R={R}: {dt/60*1000:.2f} ms per step")
# compare <m^2> at T=2.6 between nd and old implementation (should agree within stat errors)
t0 = time.time()
Mnd, _ = swnd.sw_run_nd(2.6, R=256, n_steps=400, n_burn=100, seed=5, N=16)
Mold, _ = sw.sw_run(2.6, R=256, n_steps=400, n_burn=100, seed=6, N=16)
for name, MM in [("nd", Mnd), ("old", Mold)]:
    m = MM / 256.0
    print(name, "<m2>=", (m**2).mean().round(5), "+-", ((m**2).mean(axis=0).std()/np.sqrt(MM.shape[1])).round(5),
          " U=", (1 - (m**4).mean()/(3*(m**2).mean()**2)).round(5))
print("time", round(time.time()-t0, 1))
