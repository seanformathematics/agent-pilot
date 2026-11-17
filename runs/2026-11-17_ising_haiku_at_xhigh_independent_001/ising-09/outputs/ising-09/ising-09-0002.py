import os, sys, time
os.makedirs('/work', exist_ok=True)
src = r'''
import numpy as np, sys
assert sys.byteorder == "little"

class PackedWolff:
    def __init__(self, N, W, T, seed, K=24):
        self.N = N; self.W = W; self.T = T
        self.rng = np.random.default_rng(seed)
        p = 1.0 - np.exp(-2.0 / T)
        self.p = p
        self.K = K
        self.P = int(round(p * 2**K))
        self.S = self.rng.integers(0, 2**64, size=(W, N, N), dtype=np.uint64)
        self.bitvals = [np.uint64(1) << np.uint64(b) for b in range(64)]
        self.shape = (W, N, N)

    def bern(self):
        X = np.zeros(self.shape, dtype=np.uint64)
        for k in range(self.K, 0, -1):
            R = self.rng.integers(0, 2**64, size=self.shape, dtype=np.uint64)
            if (self.P >> (self.K - k)) & 1:
                X |= R
            else:
                X &= R
        return X

    def step(self):
        N, W = self.N, self.W
        S = self.S
        Bx = ~(S ^ np.roll(S, -1, axis=1)) & self.bern()
        By = ~(S ^ np.roll(S, -1, axis=2)) & self.bern()
        C = np.zeros(self.shape, dtype=np.uint64)
        Cf = C.reshape(W, N * N)
        idx = self.rng.integers(0, N * N, size=(W, 64))
        ar = np.arange(W)
        for b in range(64):
            Cf[ar, idx[:, b]] |= self.bitvals[b]
        F = C.copy()
        while True:
            new = np.roll(F & Bx, 1, axis=1)
            new |= np.roll(F, -1, axis=1) & Bx
            new |= np.roll(F & By, 1, axis=2)
            new |= np.roll(F, -1, axis=2) & By
            new &= ~C
            if not new.any():
                break
            C |= new
            F = new
        self.S = S ^ C
        return C

    def counts(self):
        N, W = self.N, self.W
        b = self.S.view(np.uint8).reshape(W, N, N, 8)
        bits = np.unpackbits(b, axis=-1, bitorder='little')
        return bits.sum(axis=(1, 2), dtype=np.int32).reshape(W * 64)

def run(N, W, T, seed, nburn, nmeas, block=100):
    sim = PackedWolff(N, W, T, seed)
    for _ in range(nburn):
        sim.step()
    L = W * 64
    s2 = np.zeros(L); s4 = np.zeros(L)
    nb = nmeas // block
    blk2 = np.zeros((nb, L)); blk4 = np.zeros((nb, L))
    cnt = 0
    NN = N * N
    for t in range(nb):
        b2 = np.zeros(L); b4 = np.zeros(L)
        for j in range(block):
            sim.step()
            n = sim.counts()
            m = (2.0 * n - NN) / NN
            m2 = m * m
            b2 += m2; b4 += m2 * m2
        s2 += b2; s4 += b4
        blk2[t] = b2 / block; blk4[t] = b4 / block
        cnt += block
    return dict(s2=s2, s4=s4, cnt=cnt, blk2=blk2, blk4=blk4)

def binder_from_sums(s2, s4, cnt):
    m2 = s2.sum() / (len(s2) * cnt)
    m4 = s4.sum() / (len(s4) * cnt)
    return 1.0 - m4 / (3.0 * m2 * m2)

def jackknife_U(s2, s4, cnt):
    L = len(s2)
    T2 = s2.sum(); T4 = s4.sum()
    m2c = (T2 - s2) / ((L - 1) * cnt)
    m4c = (T4 - s4) / ((L - 1) * cnt)
    Uc = 1.0 - m4c / (3.0 * m2c * m2c)
    U = 1.0 - (T4 / (L * cnt)) / (3.0 * (T2 / (L * cnt)) ** 2)
    var = (L - 1) / L * np.sum((Uc - Uc.mean()) ** 2)
    return U, np.sqrt(var)
'''
open('/work/wolff.py', 'w').write(src)
sys.path.insert(0, '/work')
import numpy as np
import wolff as wf

# 1) Bernoulli bit generator check
sim = wf.PackedWolff(4, 4, 2.0, seed=1)
for T in [1.5, 2.0, 2.8]:
    sim.P = int(round((1 - np.exp(-2 / T)) * 2**sim.K))
    X = sim.bern()
    frac = np.unpackbits(X.view(np.uint8)).mean()
    print("T", T, "p", round(1 - np.exp(-2 / T), 6), "empirical", round(frac, 6))

# 2) exact enumeration for N=4
def exact_U(N, T):
    NN = N * N
    confs = np.arange(2**NN, dtype=np.uint32)
    spins = ((confs[:, None] >> np.arange(NN)) & 1).astype(np.int8) * 2 - 1
    s = spins.reshape(-1, N, N).astype(np.int64)
    E = -(s * np.roll(s, -1, axis=1)).sum(axis=(1, 2)) - (s * np.roll(s, -1, axis=2)).sum(axis=(1, 2))
    m = spins.sum(axis=1) / NN
    E = E - E.min()
    w = np.exp(-E / T)
    w /= w.sum()
    m2 = (w * m**2).sum(); m4 = (w * m**4).sum()
    return 1 - m4 / (3 * m2 * m2)

t0 = time.time()
for T in [1.5, 2.0, 2.27, 2.8]:
    Ue = exact_U(4, T)
    r = wf.run(4, 4, T, seed=12345, nburn=200, nmeas=20000, block=100)
    U, Ue_err = wf.jackknife_U(r['s2'], r['s4'], r['cnt'])
    print(f"T={T} exact U={Ue:.5f}  wolff U={U:.5f} +- {Ue_err:.5f}")
print("time", time.time() - t0)
