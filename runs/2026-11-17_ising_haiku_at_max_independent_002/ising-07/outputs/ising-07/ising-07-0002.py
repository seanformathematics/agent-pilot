import numpy as np, time, os
from scipy.special import ellipk
os.makedirs('/work', exist_ok=True)

def wolff_step(S, pb, rng, L):
    # One Wolff cluster update for every replica simultaneously.
    # S: (R,L,L) int8 spins; pb: (R,1,1) bond-activation probability 1-exp(-2/T).
    R = S.shape[0]
    HO = (rng.random((R, L, L)) < pb) & (S == np.roll(S, -1, axis=2))   # right bond open & parallel
    VO = (rng.random((R, L, L)) < pb) & (S == np.roll(S, -1, axis=1))   # down bond open & parallel
    seed = rng.integers(0, L*L, size=R)
    C = np.zeros((R, L*L), dtype=bool)
    C[np.arange(R), seed] = True
    C = C.reshape(R, L, L)
    F = C.copy()
    while True:
        new = (np.roll(F & HO, 1, axis=2) | (np.roll(F, -1, axis=2) & HO)
               | np.roll(F & VO, 1, axis=1) | (np.roll(F, -1, axis=1) & VO))
        new &= ~C
        if not new.any():
            break
        C |= new
        F = new
    return np.where(C, -S, S).astype(np.int8)

def energy_per_site(S, L):
    S = S.astype(np.int32)
    return -((S*np.roll(S, -1, axis=1)).sum(axis=(1,2)) + (S*np.roll(S, -1, axis=2)).sum(axis=(1,2))) / (L*L)

def exact_moments(L, T):
    N = L*L
    idx = np.arange(1 << N, dtype=np.int64)
    bits = (idx[:, None] >> np.arange(N)) & 1
    spins = (2*bits - 1).reshape(-1, L, L)
    E = -((spins*np.roll(spins,-1,axis=1)).sum(axis=(1,2)) + (spins*np.roll(spins,-1,axis=2)).sum(axis=(1,2)))
    m = spins.sum(axis=(1,2))/N
    w = np.exp(-(E - E.min())/T); w /= w.sum()
    return dict(m2=float((w*m**2).sum()), m4=float((w*m**4).sum()), E=float((w*E).sum()/N))

def onsager_energy(T):
    K = 1.0/T
    k = 2*np.sinh(2*K)/np.cosh(2*K)**2
    t2 = np.tanh(2*K)**2
    return float(-(1/np.tanh(2*K))*(1 + (2/np.pi)*(2*t2-1)*ellipk(k**2)))

rng = np.random.default_rng(20240601)
t0 = time.time()
print("=== validation L=4 vs exact enumeration ===")
L = 4
for T in [1.8, 2.3, 2.8]:
    ex = exact_moments(L, T)
    R = 400
    pb = np.full((R,1,1), 1-np.exp(-2.0/T))
    S = rng.choice(np.array([-1,1], dtype=np.int8), size=(R,L,L))
    nb, burn = 1500, 200
    s2 = np.zeros(R); s4 = np.zeros(R); se = np.zeros(R); cnt = 0
    for t in range(nb):
        S = wolff_step(S, pb, rng, L)
        if t >= burn:
            m = S.reshape(R,-1).sum(1)/(L*L)
            s2 += m**2; s4 += m**4; se += energy_per_site(S, L); cnt += 1
    mu2, mu4, mE = s2/cnt, s4/cnt, se/cnt
    print(f"T={T}: exact m2={ex['m2']:.5f} m4={ex['m4']:.5f} E/N={ex['E']:.5f} | "
          f"MC m2={mu2.mean():.5f}+-{mu2.std()/np.sqrt(R):.5f} m4={mu4.mean():.5f}+-{mu4.std()/np.sqrt(R):.5f} "
          f"E/N={mE.mean():.5f}+-{mE.std()/np.sqrt(R):.5f}")
print("validation time", time.time()-t0)

print("=== Onsager check (infinite-lattice energy per site) ===")
for T in [1.5, 2.0, 2.5, 2.8]:
    print(T, onsager_energy(T))

print("=== L=16 pilot ===")
L = 16
Ts = np.array([1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
nc = 64
Tr = np.repeat(Ts, nc)
R = len(Tr)
pb = (1-np.exp(-2.0/Tr)).reshape(R,1,1)
S = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,L,L))
budget = 150.0
burn = 300
mrec = []; Erec = []
ts = time.time(); t = 0
while time.time() - ts < budget:
    S = wolff_step(S, pb, rng, L)
    t += 1
    if t > burn:
        m = S.reshape(R,-1).sum(1)/(L*L)
        mrec.append(m.astype(np.float32))
        Erec.append(energy_per_site(S, L).astype(np.float32))
el = time.time()-ts
print("total steps", t, "recorded", len(mrec), "time/step (s)", el/t)
mrec = np.array(mrec); Erec = np.array(Erec)
np.savez('/work/pilot_L16.npz', m=mrec, E=Erec, Ts=Ts)
np.save('/work/pilot_state.npy', S)

def tau_int_chains(x):
    x = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    f = np.fft.rfft(x, n=2*n, axis=1)
    acf = np.fft.irfft(f*np.conj(f), axis=1)[:, :n].mean(0)
    acf = acf/acf[0]
    tau = 1.0
    for W in range(1, n//2):
        tau = 1.0 + 2.0*acf[1:W+1].sum()
        if W >= 5.0*tau:
            return tau
    return tau

def jackknife_U(m2, m4):
    nc_ = m2.shape[0]; n = m2.shape[1]
    A = m4.sum(1); B = m2.sum(1)
    Atot, Btot = A.sum(), B.sum()
    U_full = 1 - (Atot/(nc_*n))/(3*(Btot/(nc_*n))**2)
    Uj = np.empty(nc_)
    for i in range(nc_):
        Ai = Atot - A[i]; Bi = Btot - B[i]; k = nc_-1
        Uj[i] = 1 - (Ai/(k*n))/(3*(Bi/(k*n))**2)
    var = (nc_-1)/nc_*np.sum((Uj-Uj.mean())**2)
    return U_full, np.sqrt(var)

mr = mrec.reshape(mrec.shape[0], len(Ts), nc)
Er = Erec.reshape(Erec.shape[0], len(Ts), nc)
print(" T     U       Uerr    tau(m2) tau(m4) U_half1  U_half2  E/N_MC   E/N_Onsager")
for g, T in enumerate(Ts):
    x = mr[:, g, :].T.astype(np.float64)
    m2 = x**2; m4 = x**4
    U, Ue = jackknife_U(m2, m4)
    h = x.shape[1]//2
    U1,_ = jackknife_U(m2[:, :h], m4[:, :h]); U2,_ = jackknife_U(m2[:, h:], m4[:, h:])
    t2 = tau_int_chains(m2); t4 = tau_int_chains(m4)
    e = Er[:, g, :].T.astype(np.float64)
    print(f"{T:.2f} {U:.5f} {Ue:.5f} {t2:7.2f} {t4:7.2f} {U1:.5f} {U2:.5f} {e.mean():.5f} {onsager_energy(T):.5f}")
print("analysis done; total elapsed", time.time()-t0)
