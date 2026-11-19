import numpy as np, time, os
os.makedirs('/work', exist_ok=True)
code = r'''
import numpy as np, sys, time, json
N=16
def run(T, R, nequil, nmeas, seed, state=None):
    rng = np.random.Generator(np.random.PCG64(seed))
    if state is None:
        s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    else:
        s = state
    # acceptance table for dE = 2*s*nn in {-8,-4,0,4,8}; index by dE+8
    acc = np.zeros(17)
    for dE in (4,8):
        acc[dE+8] = np.exp(-dE/T)
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    masks = [((ii+jj)%2==c) for c in (0,1)]
    def sweep(s):
        for c in (0,1):
            nn = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE = 2*s*nn  # int8 in [-8,8]
            p = acc[(dE+8).astype(np.int64)]
            u = rng.random(s.shape)
            flip = masks[c][None] & ((dE<=0) | (u < p))
            s[flip] *= -1
        return s
    for _ in range(nequil):
        s = sweep(s)
    S2 = np.zeros(R); S4 = np.zeros(R); S1=np.zeros(R)
    for _ in range(nmeas):
        s = sweep(s)
        m = s.sum(axis=(1,2),dtype=np.int64)/(N*N)
        m2 = m*m
        S2 += m2; S4 += m2*m2; S1 += np.abs(m)
    return s, S2/nmeas, S4/nmeas, S1/nmeas
'''
open('/work/ising.py','w').write(code)
exec(code)
t=time.time()
s,m2,m4,a = run(2.2, 1000, 50, 50, 1)
print("time for 1000 replicas, 100 sweeps:", time.time()-t)
print(m2.mean(), m4.mean())