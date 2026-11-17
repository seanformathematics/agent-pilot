import numpy as np, time, json, os
N=16
Ts=[2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def make_tables(T):
    xs = np.array([-4,-2,0,2,4])
    p = np.where(xs<=0, 1.0, np.exp(-2.0*xs/T))
    return p.astype(np.float32)
ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks = [((ii+jj)%2==0), ((ii+jj)%2==1)]
def run(R, T, sweeps, burn, seed):
    rng = np.random.default_rng(seed)
    S = np.ones((R,N,N), dtype=np.int8)
    S[R//2:] = -1
    tab = make_tables(T)
    S2=np.zeros(R); S4=np.zeros(R); n=0
    for t in range(sweeps):
        for mk in masks:
            h = np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)
            x = S*h
            pr = tab[(x+4)>>1]
            r = rng.random((R,N,N), dtype=np.float32)
            flip = (r < pr) & mk
            np.negative(S, out=S, where=flip)
        if t>=burn:
            m = S.reshape(R,-1).mean(axis=1).astype(np.float64)
            m2=m*m
            S2+=m2; S4+=m2*m2; n+=1
    return S2/n, S4/n
os.makedirs("/work/res", exist_ok=True)
for T in Ts:
    t0=time.time()
    seed = 777000 + int(round(T*1000))
    a2,a4 = run(1000,T,11000,1000,seed)
    np.savez(f"/work/res/T{T:.2f}.npz", a2=a2, a4=a4, seed=seed)
    U = 1-a4.mean()/(3*a2.mean()**2)
    print(f"T={T} seed={seed} U={U:.5f} time={time.time()-t0:.1f}s", flush=True)
