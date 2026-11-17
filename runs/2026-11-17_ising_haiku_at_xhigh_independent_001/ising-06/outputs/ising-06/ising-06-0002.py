import numpy as np, time
N=16
def make_tables(T):
    # x = s*h in {-4,-2,0,2,4}; accept prob min(1, exp(-2x/T))
    xs = np.array([-4,-2,0,2,4])
    p = np.where(xs<=0, 1.0, np.exp(-2.0*xs/T))
    return p.astype(np.float32)
def run(R, T, sweeps, seed):
    rng = np.random.default_rng(seed)
    S = np.ones((R,N,N), dtype=np.int8)
    S[R//2:] = -1
    tab = make_tables(T)
    ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks = [((ii+jj)%2==0), ((ii+jj)%2==1)]
    out = np.zeros((sweeps,R))
    for t in range(sweeps):
        for mk in masks:
            h = np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)
            x = S*h
            pr = tab[(x+4)>>1]
            r = rng.random((R,N,N), dtype=np.float32)
            flip = (r < pr) & mk
            np.negative(S, out=S, where=flip)
        m = S.reshape(R,-1).mean(axis=1)
        out[t]=m
    return out
t0=time.time()
out = run(500, 2.269, 200, 1)
print("time per sweep (R=500):", (time.time()-t0)/200)
