
import os, textwrap
os.makedirs("/work", exist_ok=True)
src = r'''
import math, random, time
from array import array
import numpy as np

def make_neighbors(N):
    nb = []
    for i in range(N):
        for j in range(N):
            up = ((i - 1) % N) * N + j
            dn = ((i + 1) % N) * N + j
            lf = i * N + (j - 1) % N
            rt = i * N + (j + 1) % N
            nb.append((up, dn, lf, rt))
    return nb

def bonds(N):
    # right and down bonds, each bond once
    out = []
    for i in range(N):
        for j in range(N):
            a = i * N + j
            out.append((a, i * N + (j + 1) % N))
            out.append((a, ((i + 1) % N) * N + j))
    return np.array(out)

def energy_per_spin(spins_arr, bond_arr, NN):
    s = np.asarray(spins_arr, dtype=np.int64)
    return -(s[bond_arr[:, 0]] * s[bond_arr[:, 1]]).sum() / NN

def wolff_steps(spins, nb, NN, p, nsteps, rnd, rec, M):
    for _ in range(nsteps):
        i = int(rnd() * NN)
        s0 = spins[i]
        spins[i] = -s0
        stack = [i]
        pop = stack.pop
        push = stack.append
        size = 1
        while stack:
            j = pop()
            for k in nb[j]:
                if spins[k] == s0 and rnd() < p:
                    spins[k] = -s0
                    push(k)
                    size += 1
        M -= 2 * s0 * size
        if rec is not None:
            rec.append(M)
    return M

def simulate(N, T, n_burn, n_meas, seed, start="hot", init_spins=None):
    """Wolff simulation; returns array of total magnetization M (int) after each cluster update."""
    rng = random.Random(seed)
    rnd = rng.random
    NN = N * N
    nb = make_neighbors(N)
    p = 1.0 - math.exp(-2.0 / T)
    if init_spins is not None:
        spins = list(init_spins)
    elif start == "hot":
        spins = [1 if rnd() < 0.5 else -1 for _ in range(NN)]
    else:
        spins = [1] * NN
    M = sum(spins)
    M = wolff_steps(spins, nb, NN, p, n_burn, rnd, None, M)
    rec = array("i")
    M = wolff_steps(spins, nb, NN, p, n_meas, rnd, rec, M)
    return np.frombuffer(rec, dtype=np.int32).astype(np.int64), spins

def binder_from_M(M_arr, NN):
    m = np.asarray(M_arr, dtype=np.float64) / NN
    m2 = m * m
    m4 = m2 * m2
    return 1.0 - m4.mean() / (3.0 * m2.mean() ** 2)

def binder_jackknife(M_arr, NN, nblocks=200):
    m = np.asarray(M_arr, dtype=np.float64) / NN
    m2 = m * m
    m4 = m2 * m2
    n = len(m)
    B = nblocks
    L = n // B
    m2 = m2[: B * L].reshape(B, L)
    m4 = m4[: B * L].reshape(B, L)
    S2 = m2.sum(axis=1)
    S4 = m4.sum(axis=1)
    T2 = S2.sum()
    T4 = S4.sum()
    tot = B * L
    s2 = (T2 - S2) / ((B - 1) * L)
    s4 = (T4 - S4) / ((B - 1) * L)
    Uj = 1.0 - s4 / (3.0 * s2 * s2)
    Ufull = 1.0 - (T4 / tot) / (3.0 * (T2 / tot) ** 2)
    Ubar = Uj.mean()
    err = math.sqrt((B - 1) / B * np.sum((Uj - Ubar) ** 2))
    return Ufull, err
'''
with open("/work/ising.py", "w") as f:
    f.write(src)

import sys, time
sys.path.insert(0, "/work")
import importlib
import ising
importlib.reload(ising)
import numpy as np, math, itertools

# ---- exact enumeration on 4x4 ----
N = 4; NN = 16
configs = ((np.arange(2**NN)[:, None] >> np.arange(NN)[None, :]) & 1) * 2 - 1
bd = ising.bonds(N)
E = -(configs[:, bd[:, 0]] * configs[:, bd[:, 1]]).sum(axis=1)  # total energy
Mtot = configs.sum(axis=1)
exact = {}
for T in [2.0, 2.5, 3.0]:
    w = np.exp(-(E - E.min()) / T)
    Z = w.sum()
    m = Mtot / NN
    m2 = (w * m**2).sum() / Z
    m4 = (w * m**4).sum() / Z
    U = 1 - m4 / (3 * m2**2)
    e = (w * E).sum() / Z / NN
    exact[T] = (U, m2, e)
    print(f"exact 4x4 T={T}: U={U:.6f} <m^2>={m2:.6f} e/spin={e:.6f}")

for T in [2.0, 2.5, 3.0]:
    t0 = time.time()
    Mrec, spins = ising.simulate(N, T, 2000, 400000, seed=12345 + int(T*10), start="cold")
    Uest, Uerr = ising.binder_jackknife(Mrec, NN, nblocks=100)
    m2 = np.mean((Mrec / NN) ** 2)
    print(f"wolff 4x4 T={T}: U={Uest:.5f} +- {Uerr:.5f}  <m^2>={m2:.5f} (exact {exact[T][1]:.5f}) exact U={exact[T][0]:.5f} time={time.time()-t0:.1f}s")

# ---- timing on 16x16 ----
N = 16; NN = 256
t0 = time.time()
Mrec, spins = ising.simulate(N, 2.269, 2000, 20000, seed=7, start="cold")
dt = time.time() - t0
print(f"16x16 T=2.269: 22000 clusters in {dt:.2f}s -> {dt/22000*1e6:.1f} us per cluster")
print("mean |m|", np.mean(np.abs(Mrec))/NN, "U", ising.binder_from_M(Mrec, NN))
