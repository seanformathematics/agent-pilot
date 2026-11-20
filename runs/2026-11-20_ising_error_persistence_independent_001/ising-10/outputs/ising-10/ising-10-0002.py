
import numpy as np, time, os
os.makedirs('/work', exist_ok=True)
code = r'''
import numpy as np, time, sys, json
def run(T, N=16, R=2000, seed=1, n_therm=1000, t_budget=100.0, nsave_max=None):
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    masks = [((ii+jj)%2==p) for p in (0,1)]
    beta = 1.0/T
    # acceptance table for flip: dE = 2*s*nn, nn in {-4,-2,0,2,4}
    # index by (s*nn+4)//2 -> values 0..4 for s*nn in {-4,-2,0,2,4}
    acc = np.zeros((2,5), dtype=np.float64)
    for k in range(5):
        snn = 2*k-4
        acc[0,k] = np.exp(-beta*2*snn) if snn>0 else 1.0  # placeholder fixed below
    acc = np.array([[min(1.0, np.exp(-beta*2*(2*k-4))) for k in range(5)]])
    thr = acc[0]
    sums = []
    t0 = time.time()
    sweep = 0
    m_acc = np.zeros(R); m2_acc=np.zeros(R); m4_acc=np.zeros(R); cnt=0
    while True:
        for p in (0,1):
            nn = (np.roll(s,1,1).astype(np.int8)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            # flip prob for s with neighbor sum nn: dE=2*s*nn, index=(s*nn+4)//2
            idx = ((s*nn).astype(np.int16)+4)//2
            pr = thr[idx]
            u = rng.random(size=(R,N,N))
            flip = (u < pr) & masks[p]
            s = np.where(flip, -s, s).astype(np.int8)
        sweep += 1
        if sweep > n_therm:
            m = np.abs(s.reshape(R,-1).mean(axis=1))
            m2 = m*m
            m2_acc += m2; m4_acc += m2*m2; cnt += 1
            # save per-sweep replica-averaged quantities (time series over sweeps)
            sums.append((m2.mean(), (m2*m2).mean()))
        if time.time()-t0 > t_budget: break
    return sweep, cnt, m2_acc/cnt, m4_acc/cnt, np.array(sums)
'''
open('/work/ising.py','w').write(code)
exec(code)
t=time.time()
sw, cnt, m2r, m4r, ts = run(2.269, R=2000, seed=1, n_therm=10, t_budget=20)
print("sweeps", sw, "time", time.time()-t, "ms/sweep", 1000*(time.time()-t)/sw)
