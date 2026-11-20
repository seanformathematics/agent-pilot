
import numpy as np, time
def run(T, R, nwarm, nmeas, seed):
    rng = np.random.default_rng(seed)
    N=16
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks = [((ii+jj)%2==0), ((ii+jj)%2==1)]
    beta=1.0/T
    acc = {d: np.exp(-beta*d) for d in (4,8)}
    def sweep(s):
        for m in masks:
            nn = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE = 2*s*nn  # int8 ok
            p = np.where(dE<=0, 1.0, np.where(dE==4, acc[4], np.where(dE==8, acc[8], 0.0)))
            flip = (rng.random(s.shape) < p) & m
            s = np.where(flip, -s, s).astype(np.int8)
        return s
    for _ in range(nwarm):
        s = sweep(s)
    M2=np.zeros(R); M4=np.zeros(R); cnt=0
    t0=time.time()
    for _ in range(nmeas):
        s = sweep(s)
        m = s.reshape(R,-1).mean(axis=1, dtype=np.float64)
        M2 += m*m; M4 += m**4; cnt+=1
    return M2/cnt, M4/cnt, time.time()-t0
t=time.time()
M2,M4,dt = run(2.269, 1000, 10, 50, 1)
print("time for 60 sweeps R=1000:", dt, "total", time.time()-t)
