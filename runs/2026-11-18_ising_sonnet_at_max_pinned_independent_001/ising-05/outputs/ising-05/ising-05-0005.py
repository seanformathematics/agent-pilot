
import sys, os, time
os.makedirs('/work', exist_ok=True)

kauf = r'''
import numpy as np
def logZ_kaufman(K, N):
    l = np.arange(2 * N)
    cosh_g = np.cosh(2 * K) ** 2 / np.sinh(2 * K) - np.cos(np.pi * l / N)
    g = np.arccosh(cosh_g)
    g[0] = 2 * K + np.log(np.tanh(K))
    godd = g[1::2]; geven = g[0::2]
    def lc(x):
        return np.sum(np.log(2 * np.cosh(N * x / 2)))
    def ls(x):
        v = 2 * np.sinh(N * x / 2)
        return np.sum(np.log(np.abs(v))), np.prod(np.sign(v))
    z1 = lc(godd); z2, s2 = ls(godd); z3 = lc(geven); z4, s4 = ls(geven)
    terms = np.array([z1, z2, z3, z4]); signs = np.array([1.0, s2, 1.0, s4])
    m = terms.max()
    tot = np.sum(signs * np.exp(terms - m))
    return N * N / 2 * np.log(2 * np.sinh(2 * K)) + np.log(0.5) + m + np.log(tot)

def e_exact(T, N, h=1e-5):
    K = 1.0 / T
    return -(logZ_kaufman(K + h, N) - logZ_kaufman(K - h, N)) / (2 * h) / (N * N)
'''
with open('/work/kauf.py', 'w') as f:
    f.write(kauf)

prod = r'''
import sys, time
sys.path.insert(0, '/work')
import numpy as np
import sw, kauf

TS = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
SEEDS = [7000 + i for i in range(len(TS))]
N = 16

def jk_mean(x, G):
    R = len(x); k = R // G
    xg = x.reshape(G, k).sum(axis=1)
    xj = (x.sum() - xg) / (R - k)
    return x.mean(), np.sqrt((G - 1) / G * np.sum((xj - xj.mean()) ** 2))

def run_T(i, R=1000, neq=300, nmeas=2000, blk=100):
    T = TS[i]; seed = SEEDS[i]
    sim = sw.SwendsenWang(N, R, T, seed, start="hot")
    sim.s[:R // 2] = 1          # first half of the replicas: cold (ordered) start, second half: hot (random) start
    for _ in range(neq):
        sim.sweep(measure=False)
    keys = ["m2", "m4", "m2i", "m4i", "e"]
    nb = nmeas // blk
    blocks = {k: np.zeros((R, nb)) for k in keys}
    t0 = time.time()
    for b in range(nb):
        acc = {k: np.zeros(R) for k in keys}
        for _ in range(blk):
            o = sim.sweep(measure=True, energy=True)
            for k in keys:
                acc[k] += o[k]
        for k in keys:
            blocks[k][:, b] = acc[k] / blk
    dt = time.time() - t0
    np.savez(f'/work/prod_T{T}.npz', T=T, seed=seed, R=R, neq=neq, nmeas=nmeas, blk=blk, **blocks)
    return blocks, dt

def report(i, blocks):
    T = TS[i]
    R = blocks["m2"].shape[0]
    pr = {k: blocks[k].mean(axis=1) for k in blocks}
    U1, e1 = sw.jackknife_U(pr["m2"], pr["m4"], 100)
    U2, e2 = sw.jackknife_U(pr["m2i"], pr["m4i"], 100)
    # hot/cold halves
    Uc, ec = sw.jackknife_U(pr["m2"][:R // 2], pr["m4"][:R // 2], 50)
    Uh, eh = sw.jackknife_U(pr["m2"][R // 2:], pr["m4"][R // 2:], 50)
    zhc = (Uc - Uh) / np.hypot(ec, eh)
    em, eerr = jk_mean(pr["e"], 100)
    ex = kauf.e_exact(T, N)
    print(f"T={T}: U_plain={U1:.5f}+-{e1:.5f}  U_impr={U2:.5f}+-{e2:.5f}  cold/hot z={zhc:+.2f} (Uc={Uc:.5f}, Uh={Uh:.5f})  e={em:.5f}+-{eerr:.5f} exact={ex:.5f} z={(em-ex)/eerr:+.2f}", flush=True)

def main(idx, tmax=520):
    t00 = time.time()
    for i in idx:
        if time.time() - t00 > tmax - 45:
            print("stopping early before T index", i, flush=True)
            break
        blocks, dt = run_T(i)
        print(f"[{time.time()-t00:.0f}s, meas {dt:.0f}s]", end=" ")
        report(i, blocks)
'''
with open('/work/prod.py', 'w') as f:
    f.write(prod)

sys.path.insert(0, '/work')
import prod
prod.main([0, 1, 2, 3, 4, 5, 6])
