src = '''
import math, random
import numpy as np

def neighbors(N):
    nbr = []
    for y in range(N):
        for x in range(N):
            nbr.append((y*N + (x+1) % N, y*N + (x-1) % N, ((y+1) % N)*N + x, ((y-1) % N)*N + x))
    return nbr

def run_chain(N, T, seed, n_equil, n_meas, init=0):
    NN = N*N
    nbr = neighbors(N)
    p = 1.0 - math.exp(-2.0/T)
    rnd = random.Random(seed).random
    if init == 0:
        spins = [1 if rnd() < 0.5 else -1 for _ in range(NN)]
    else:
        spins = [1]*NN
    M = sum(spins)
    out = np.empty(n_meas, dtype=np.int16)
    k = 0
    for it in range(n_equil + n_meas):
        ss = int(rnd()*NN)
        s0 = spins[ss]
        spins[ss] = -s0
        stack = [ss]
        push = stack.append
        pop = stack.pop
        csize = 1
        while stack:
            i = pop()
            for j in nbr[i]:
                if spins[j] == s0 and p > rnd():
                    spins[j] = -s0
                    push(j)
                    csize += 1
        M -= 2*s0*csize
        if it > n_equil - 1:
            out[k] = M
            k += 1
    return out, spins

def binder_jk(M, bin_size):
    M = np.asarray(M, dtype=np.float64)
    m2 = M*M
    m4 = m2*m2
    n = len(M)
    K = n // bin_size
    nt = K*bin_size
    m2 = m2[:nt].reshape(K, bin_size)
    m4 = m4[:nt].reshape(K, bin_size)
    s2 = m2.sum(axis=1)
    s4 = m4.sum(axis=1)
    S2 = s2.sum()
    S4 = s4.sum()
    U_full = 1.0 - (S4/nt)/(3.0*(S2/nt)**2)
    n_k = nt - bin_size
    S2k = S2 - s2
    S4k = S4 - s4
    Uk = 1.0 - (S4k/n_k)/(3.0*(S2k/n_k)**2)
    Ubar = Uk.mean()
    err = math.sqrt((K-1)/K * np.sum((Uk-Ubar)**2))
    return U_full, err, K

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
        if t - c*tau > -1e-12:
            break
    return tau
'''
with open("/work/wolff.py", "w") as f:
    f.write(src)
import sys
sys.path.insert(0, "/work")
import wolff
M, _ = wolff.run_chain(4, 2.0, seed=1, n_equil=100, n_meas=1000)
print("module ok; sample M values:", M[:10], "len", len(M))
