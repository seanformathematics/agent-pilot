
code = r'''
import numpy as np

def exact_ZM(N, K):
    """Exact Z_M = sum over configs with total magnetization M of exp(K * sum_<ij> s_i s_j) for the N x N torus.
    Magnetization-resolved transfer matrix over rows, trace over the initial row. Returns M values and Z_M
    (scaled by a constant)."""
    S = 1 << N
    states = np.arange(S)
    bits = ((states[:, None] >> np.arange(N)[None, :]) & 1)
    spins = 2 * bits - 1                                   # (S,N)
    mrow = spins.sum(axis=1)                               # row magnetization
    hb = (spins * np.roll(spins, -1, axis=1)).sum(axis=1)  # horizontal bonds within a row (periodic)
    wh = np.exp(K * hb)
    V = np.exp(K * (spins @ spins.T))                      # vertical bond weights between rows s and s'
    nM = N * N + 1                                         # M index: (M + N*N)/2 in 0..N*N
    Z = np.zeros(nM)
    shift = (mrow + N) // 2                                # row magnetization in units of 2, shifted to >=0
    chunk = max(1, min(S, 2_000_000 // (S * nM // 64 + 1)))
    chunk = min(S, 64)
    for r0s in range(0, S, chunk):
        r0 = np.arange(r0s, min(S, r0s + chunk))
        nr = len(r0)
        # v[r, M, s]: weight of partial configurations starting at row r0[r], currently at row s with total M
        v = np.zeros((nr, nM, S))
        for a, r in enumerate(r0):
            v[a, (mrow[r] + N * N) // 2 - (N * N - N) // 2 * 0 - 0, r] = 0  # placeholder (overwritten below)
        v[:] = 0
        # M index convention: idx = (M + N*N)/2 ; single row with magnetization mr -> idx = (mr + N*N)/2
        for a, r in enumerate(r0):
            v[a, (mrow[r] + N * N) // 2, r] = wh[r]
        for step in range(N - 1):
            # apply vertical transfer:  u[r, M, s'] = sum_s v[r, M, s] V[s, s'] ; then multiply wh[s'] and shift M by mrow[s']
            u = (v.reshape(nr * nM, S) @ V).reshape(nr, nM, S) * wh[None, None, :]
            v = np.zeros_like(u)
            for sp in range(S):
                sh = int(mrow[sp] // 2)    # shift in index units (mrow even/odd parity handled: mrow has parity of N)
                # idx = (M+N*N)/2 ; adding mrow[sp] to M adds mrow[sp]/2 to idx -- may be half-integer if N odd; N even here
                if sh > 0:
                    v[:, sh:, sp] = u[:, :nM - sh, sp]
                elif sh < 0:
                    v[:, :nM + sh, sp] = u[:, -sh:, sp]
                else:
                    v[:, :, sp] = u[:, :, sp]
        # close the torus: vertical bond between last row s and initial row r0
        for a, r in enumerate(r0):
            Z += v[a] @ V[:, r]
    Ms = 2 * np.arange(nM) - N * N
    return Ms, Z

def moments_from_ZM(N, K):
    Ms, Z = exact_ZM(N, K)
    m = Ms / float(N * N)
    w = Z / Z.sum()
    m2 = (w * m ** 2).sum(); m4 = (w * m ** 4).sum()
    return m2, m4, 1 - m4 / (3 * m2 ** 2)
'''
open('/work/exactM.py','w').write(code)
import sys, time
sys.path.insert(0,'/work')
import numpy as np, exactM
# validate vs brute force at N=4
N=4; n=16
states = np.arange(2**n, dtype=np.uint32)
bits = ((states[:,None] >> np.arange(n,dtype=np.uint32)[None,:]) & 1).astype(np.int8)
s = (2*bits-1).reshape(-1,N,N).astype(np.int16)
E = -(s*np.roll(s,-1,axis=2) + s*np.roll(s,-1,axis=1)).sum(axis=(1,2)).astype(np.float64)
mm = s.sum(axis=(1,2))/float(n)
for T in (2.27, 2.8):
    w=np.exp(-(E-E.min())/T); Z=w.sum(); m2e=(w*mm**2).sum()/Z; m4e=(w*mm**4).sum()/Z
    print(f"N=4 T={T}: brute m2={m2e:.10f} m4={m4e:.10f} U={1-m4e/(3*m2e**2):.10f}")
    print(f"           transfer m2,m4,U = {exactM.moments_from_ZM(4,1/T)}")
t=time.time()
for T in (2.5,2.8):
    print(f"N=6 T={T}: m2,m4,U =", exactM.moments_from_ZM(6,1/T), f"({time.time()-t:.1f}s)")
