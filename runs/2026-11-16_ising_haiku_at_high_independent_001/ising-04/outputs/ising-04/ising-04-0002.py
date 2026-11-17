import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
code = r'''
import numpy as np
def run(T, N, R, nburn, nmeas, seed, thin=1):
    rng = np.random.default_rng(seed)
    s = np.where(rng.random((R,N,N))<0.5, -1, 1).astype(np.int8)
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    maskA = ((ii+jj)%2==0)[None]
    maskB = ~maskA
    # acceptance table indexed by s*nn+4 : min(1, exp(-2*s*nn/T))
    x = np.arange(-4,5)
    acc = np.minimum(1.0, np.exp(-2.0*x/T)).astype(np.float32)
    def half(s, mask):
        nn = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        sx = (s*nn).astype(np.int8)
        p = acc[sx+4]
        r = rng.random(s.shape, dtype=np.float32)
        flip = (r < p) & mask
        s = np.where(flip, -s, s).astype(np.int8)
        return s
    def sweep(s):
        s = half(s, maskA)
        s = half(s, maskB)
        return s
    for _ in range(nburn):
        s = sweep(s)
    m2 = np.zeros((nmeas//thin, R)); m4 = np.zeros((nmeas//thin, R))
    k=0
    for t in range(nmeas):
        s = sweep(s)
        if t % thin == 0:
            m = s.reshape(R,-1).mean(axis=1, dtype=np.float64)
            m2[k] = m*m; m4[k] = m**4; k+=1
    return m2, m4
def binder_jk(m2, m4):
    # m2,m4: (T_samples, R). Per-replica averages then jackknife over replicas
    a2 = m2.mean(0); a4 = m4.mean(0)
    R = a2.size
    def U(idx):
        return 1 - a4[idx].mean()/(3*a2[idx].mean()**2)
    full = U(np.arange(R))
    jk = np.empty(R)
    allidx = np.arange(R)
    S2 = a2.sum(); S4 = a4.sum()
    for r in range(R):
        s2 = (S2-a2[r])/(R-1); s4 = (S4-a4[r])/(R-1)
        jk[r] = 1 - s4/(3*s2*s2)
    err = np.sqrt((R-1)/R * np.sum((jk-jk.mean())**2))
    return full, err
'''
open('/work/ising.py','w').write(code)
import sys; sys.path.insert(0,'/work')
from ising import run, binder_jk
t=time.time()
m2,m4 = run(2.269, 16, 1000, 200, 300, seed=1)
print("time", time.time()-t)
print(binder_jk(m2,m4))
# autocorrelation of m2 averaged over replicas
x = m2.mean(1)-m2.mean()
ac = [np.dot(x[:-l],x[l:])/np.dot(x,x) for l in range(1,60,5)]
print(np.round(ac,3))
