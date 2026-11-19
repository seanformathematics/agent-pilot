
code = r'''
import numpy as np

def metropolis_run(N, R, T, seed, n_equil, n_meas, start="hot", B=100):
    """Checkerboard single-spin-flip Metropolis, R independent replicas.
    Returns per-replica block means (block B sweeps) of m^2, m^4 and energy/site (energy every sweep of last
    sweep in block only -> cheap)."""
    rng = np.random.default_rng(seed)
    if start == "hot":
        s = (2 * rng.integers(0, 2, (R, N, N), dtype=np.int8) - 1).astype(np.int8)
    else:
        s = np.ones((R, N, N), dtype=np.int8)
    acc = np.array([min(1.0, np.exp(-2.0 * k / T)) for k in range(-4, 5)], dtype=np.float32)
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    mask = [((ii + jj) % 2 == c)[None] for c in (0, 1)]
    def sweep(s):
        for c in (0, 1):
            nn = np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2)
            k = s * nn
            p = acc[k + 4]
            u = rng.random((R, N, N), dtype=np.float32)
            flip = (u < p) & mask[c]
            s = np.where(flip, -s, s)
        return s
    for t in range(n_equil):
        s = sweep(s)
    nb = n_meas // B
    blk = np.zeros((2, R, nb))
    for b in range(nb):
        a2 = np.zeros(R); a4 = np.zeros(R)
        for t in range(B):
            s = sweep(s)
            m = s.sum(axis=(1, 2), dtype=np.int32) / float(N * N)
            m2 = m * m
            a2 += m2; a4 += m2 * m2
        blk[0, :, b] = a2 / B; blk[1, :, b] = a4 / B
    return dict(blk=blk, N=N, R=R, T=T, seed=seed, n_equil=n_equil, n_meas=nb * B, B=B, start=start)
'''
open('/work/metro.py','w').write(code)
import sys, time
sys.path.insert(0,'/work')
import numpy as np, metro
for R in (500,1000):
    t=time.time(); r=metro.metropolis_run(16,R,2.8,seed=1,n_equil=20,n_meas=100,B=50); dt=time.time()-t
    print(f"R={R}: {dt/(120)*1e3:.2f} ms/sweep -> {dt/120/R*1e6:.1f} us per replica-sweep")
# quick sanity: N=4 against exact at T=2.27 
N=4; n=16
states = np.arange(2**n, dtype=np.uint32)
bits = ((states[:,None] >> np.arange(n,dtype=np.uint32)[None,:]) & 1).astype(np.int8)
s = (2*bits-1).reshape(-1,N,N).astype(np.int16)
E = -(s*np.roll(s,-1,axis=2) + s*np.roll(s,-1,axis=1)).sum(axis=(1,2)).astype(np.float64)
mm = s.sum(axis=(1,2))/float(n)
T=2.27
w=np.exp(-(E-E.min())/T); Z=w.sum(); m2e=(w*mm**2).sum()/Z; m4e=(w*mm**4).sum()/Z
r=metro.metropolis_run(4,1000,T,seed=5,n_equil=200,n_meas=5000,B=100)
import sw
s2=r['blk'][0].mean(axis=1); s4=r['blk'][1].mean(axis=1)
U,e=sw.U_jack(s2,s4)
print("N=4 T=2.27 Metropolis U=%.5f +- %.5f ; exact U=%.5f"%(U,e,1-m4e/(3*m2e**2)))
