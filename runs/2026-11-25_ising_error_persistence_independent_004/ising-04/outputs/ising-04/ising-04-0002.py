
import numpy as np, time, os
os.makedirs("/work", exist_ok=True)
N=16

def run(T, R, n_eq, n_meas, every, seed):
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    # acceptance table indexed by dE/4+2 for dE in {-8,-4,0,4,8}
    acc = {dE: np.exp(-dE/T) for dE in (4,8)}
    ii, jj = np.indices((N,N))
    masks = [((ii+jj)%2==p) for p in (0,1)]
    def sweep(s):
        for p in (0,1):
            nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE = 2*s*nb  # in {-8,-4,0,4,8}
            u = rng.random(s.shape)
            flip = masks[p] & ((dE<=0) | (u < np.where(dE==4, acc[4], acc[8])))
            s = np.where(flip, -s, s).astype(np.int8)
        return s
    for _ in range(n_eq):
        s = sweep(s)
    m2s=[]; m4s=[]
    nm = n_meas
    M2 = np.zeros(R); M4=np.zeros(R); cnt=0
    for k in range(nm*every):
        s = sweep(s)
        if (k+1)%every==0:
            m = s.reshape(R,-1).mean(axis=1, dtype=np.float64)
            M2 += m*m; M4 += m**4; cnt+=1
    return M2/cnt, M4/cnt

t=time.time()
M2,M4 = run(2.269, 400, 200, 20, 5, 1)
print("time", time.time()-t)
print("U est", np.mean(1-M4/(3*M2**2)))
