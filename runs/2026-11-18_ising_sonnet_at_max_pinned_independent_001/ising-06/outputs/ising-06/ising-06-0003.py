
import sys, time
sys.path.insert(0,'/work')
import numpy as np, sw

# exact enumeration 4x4
N=4
nst = 1<<(N*N)
states = np.arange(nst, dtype=np.uint32)
bits = ((states[:,None] >> np.arange(N*N, dtype=np.uint32)[None,:]) & 1).astype(np.int8)
spins = (2*bits-1).reshape(nst, N, N)
M = spins.sum(axis=(1,2)).astype(np.float64)
E = -(spins*(np.roll(spins,-1,axis=2)+np.roll(spins,-1,axis=1))).sum(axis=(1,2)).astype(np.float64)
def exact(T):
    w = np.exp(-(E-E.min())/T)
    Z = w.sum()
    m2 = (w*(M/N**2)**2).sum()/Z
    m4 = (w*(M/N**2)**4).sum()/Z
    e = (w*E).sum()/Z/N**2
    return m2, m4, 1-m4/(3*m2**2), e

for T in [1.5, 2.3, 2.8]:
    ex = exact(T)
    R=2000
    b = sw.SWBatch(N, R, T, seed=12345, start="hot")
    for _ in range(100): b.step()
    nmeas=1500
    acc = dict(m2=0.0,m4=0.0,i2=0.0,i4=0.0,e=0.0)
    for _ in range(nmeas):
        S2,S4 = b.step()
        m,e = b.observables()
        acc['m2'] += np.mean(m**2); acc['m4'] += np.mean(m**4)
        acc['e'] += np.mean(e)
        acc['i2'] += np.mean(S2/N**4); acc['i4'] += np.mean((3*S2**2-2*S4)/N**8)
    for k in acc: acc[k]/=nmeas
    print(f"T={T}")
    print("  exact : m2=%.5f m4=%.5f U=%.5f e=%.5f" % ex)
    print("  plain : m2=%.5f m4=%.5f U=%.5f e=%.5f" % (acc['m2'],acc['m4'],1-acc['m4']/(3*acc['m2']**2),acc['e']))
    print("  improv: m2=%.5f m4=%.5f U=%.5f" % (acc['i2'],acc['i4'],1-acc['i4']/(3*acc['i2']**2)))
