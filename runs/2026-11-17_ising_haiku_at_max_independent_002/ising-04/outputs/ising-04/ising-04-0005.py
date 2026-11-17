import sys, time
sys.path.insert(0, "/work")
import numpy as np
import wolff

def reference_U(N, Ts):
    NN = N*N
    nconf = 2**NN
    idx = np.arange(nconf, dtype=np.int64)
    spins = np.empty((nconf, NN), dtype=np.int64)
    for k in range(NN):
        spins[:, k] = 2*((idx // (2**k)) % 2) - 1
    E = np.zeros(nconf, dtype=np.int64)
    for y in range(N):
        for x in range(N):
            i = y*N + x
            right = y*N + (x+1) % N
            down = ((y+1) % N)*N + x
            E = E - spins[:, i]*spins[:, right] - spins[:, i]*spins[:, down]
    M = spins.sum(axis=1).astype(np.float64)
    out = {}
    for T in Ts:
        w = np.exp(-(E - E.min())/T)
        Z = w.sum()
        m2 = (w*M**2).sum()/Z
        m4 = (w*M**4).sum()/Z
        out[T] = 1 - m4/(3*m2**2)
    return out

Ts = [1.5, 2.0, 2.269, 2.6, 3.5]
ref = reference_U(4, Ts)
print("reference:", [round(float(ref[T]), 6) for T in Ts])
t0 = time.time()
for T in Ts:
    M, _ = wolff.run_chain(4, T, seed=12345, n_equil=2000, n_meas=200000, init=0)
    U, err, K = wolff.binder_jk(M, 2000)
    tau = wolff.tau_int(M.astype(float)**2)
    print("T", T, "U", round(U, 5), "err", round(err, 5), "ref", round(float(ref[T]), 5), "pull", round((U-ref[T])/err, 2), "tau", round(tau, 2))
print("check time", round(time.time()-t0, 1), "s")
t0 = time.time()
M, _ = wolff.run_chain(16, 2.269, seed=777, n_equil=1000, n_meas=20000, init=0)
dt = time.time()-t0
print("L16 updates per s", round(21000/dt))
print("quick U", wolff.binder_jk(M, 1000)[0], "mean m2", np.mean((M.astype(float)/256)**2))
