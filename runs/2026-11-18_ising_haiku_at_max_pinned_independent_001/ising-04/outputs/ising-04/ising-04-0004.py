
import sys, math, random, time
sys.path.insert(0, "/work")
import numpy as np
import ising
from scipy.special import ellipk

def onsager_energy(T):
    K = 1.0 / T
    th2 = math.tanh(2*K); sh = math.sinh(2*K); ch = math.cosh(2*K)
    k = 2*sh/ch**2
    return -(ch/sh) * (1 + (2/math.pi)*(2*th2**2 - 1)*ellipk(k*k))

for T in [2.8, 2.269, 2.0, 1.5]:
    print("Onsager e/spin at T=%.3f: %.5f" % (T, onsager_energy(T)))

# Wolff with bookkeeping check on 16x16 at T=2.8
N = 16; NN = N*N
T = 2.8
rng = random.Random(42); rnd = rng.random
nb = ising.make_neighbors(N)
p = 1 - math.exp(-2/T)
spins = [1 if rnd() < 0.5 else -1 for _ in range(NN)]
M = sum(spins)
bd = ising.bonds(N)
Ms = []; Es = []; nn = []
for blk in range(400):
    M = ising.wolff_steps(spins, nb, NN, p, 50, rnd, None, M)
    assert M == sum(spins), ("bookkeeping mismatch", M, sum(spins))
    s = np.array(spins)
    Es.append(-(s[bd[:,0]]*s[bd[:,1]]).sum()/NN)
    Ms.append(M)
Ms = np.array(Ms)
print("Wolff T=2.8: mean e/spin = %.4f, mean m^2 = %.5f, U(block) = %.4f" % (np.mean(Es), np.mean((Ms/NN)**2), 1-np.mean((Ms/NN)**4)/(3*np.mean((Ms/NN)**2)**2)))

# Independent single-spin Metropolis (pure python, checkerboard-free sequential) at T=2.8
def metropolis_sweeps(spins, nb, nsw, beta, rnd):
    for _ in range(nsw):
        for i in range(NN):
            nsum = spins[nb[i][0]] + spins[nb[i][1]] + spins[nb[i][2]] + spins[nb[i][3]]
            dE = 2 * spins[i] * nsum
            if dE <= 0 or rnd() < math.exp(-beta*dE):
                spins[i] = -spins[i]
    return spins

rng2 = random.Random(7); rnd2 = rng2.random
spins2 = [1 if rnd2() < 0.5 else -1 for _ in range(NN)]
beta = 1/T
t0 = time.time()
metropolis_sweeps(spins2, nb, 300, beta, rnd2)
Mm = []; Em = []
for sw in range(3000):
    metropolis_sweeps(spins2, nb, 1, beta, rnd2)
    s = np.array(spins2)
    Mm.append(s.sum()); Em.append(-(s[bd[:,0]]*s[bd[:,1]]).sum()/NN)
Mm = np.array(Mm)
print("Metropolis T=2.8: mean e/spin = %.4f, mean m^2 = %.5f, <|m|> = %.4f, time %.1fs" % (np.mean(Em), np.mean((Mm/NN)**2), np.mean(np.abs(Mm))/NN, time.time()-t0))
