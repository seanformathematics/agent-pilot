
import numpy as np, time, os
os.makedirs("/work", exist_ok=True)
code = r'''
import numpy as np, time, sys, json
def run(T, R, N, nburn, nmeas, nsep, seed):
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    # acceptance table indexed by dE+8 for dE in {-8,-4,0,4,8}
    acc = np.zeros(17, dtype=np.float32)
    for dE in (4,8):
        acc[dE+8] = np.exp(-dE/T)
    acc[8-4] = 0  # placeholder, will be overwritten below
    acc[:] = 0
    for dE in (-8,-4,0,4,8):
        acc[dE+8] = min(1.0, np.exp(-dE/T))
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    masks = [((ii+jj)%2==p) for p in (0,1)]
    def sweep(s):
        for p in (0,1):
            nn = (np.roll(s,1,axis=1)+np.roll(s,-1,axis=1)+np.roll(s,1,axis=2)+np.roll(s,-1,axis=2)).astype(np.int8)
            dE = (2*s*nn).astype(np.int8)  # in {-8,-4,0,4,8}
            pr = acc[dE.astype(np.int16)+8]
            u = rng.random(s.shape, dtype=np.float32)
            flip = (u < pr) & masks[p][None]
            s = np.where(flip, -s, s).astype(np.int8)
        return s
    for _ in range(nburn):
        s = sweep(s)
    S2 = np.zeros(R); S4 = np.zeros(R); E1=np.zeros(R); nm=0
    for k in range(nmeas):
        for _ in range(nsep):
            s = sweep(s)
        m = s.reshape(R,-1).mean(axis=1, dtype=np.float64)
        m2 = m*m
        S2 += m2; S4 += m2*m2; nm += 1
    return S2/nm, S4/nm
'''
open("/work/ising.py","w").write(code)
exec(code)
t=time.time()
for R in (256,1024):
    t=time.time()
    a,b = run(2.0, R, 16, 0, 10, 1, 1)
    dt=time.time()-t
    print(R, dt, "per sweep ms", dt/ (10) *1000/1 , "per replica-sweep us", dt/(10*R)*1e6)
