import numpy as np, time, os, sys
os.makedirs('/work',exist_ok=True)
N=16; R=1000
Ts=[float(x) for x in sys.argv[1].split(',')] if len(sys.argv)>1 else [1.5,1.6,1.7,1.8,1.9,2.0,2.1]
EQ=2000; MEAS=8000; EVERY=5
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def run(T,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    # precomputed acceptance for dE=4,8
    p4=np.exp(-4/T); p8=np.exp(-8/T)
    def sweep(s):
        for p in (0,1):
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*h
            u=rng.random(s.shape)
            acc=(dE<=0)|((dE==4)&(u<p4))|((dE==8)&(u<p8))
            flip=acc&masks[p]
            s=np.where(flip,-s,s)
        return s
    for _ in range(EQ): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); n=0
    for k in range(MEAS):
        s=sweep(s)
        if (k+1)%EVERY==0:
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2=m*m
            S2+=m2; S4+=m2*m2; n+=1
    return S2/n, S4/n
for T in Ts:
    t0=time.time()
    seed=int(round(T*1000))+7
    m2,m4=run(T,seed)
    np.savez(f'/work/res_N{N}_T{T:.2f}.npz',m2=m2,m4=m4,seed=seed,T=T)
    U=1-m4.mean()/(3*m2.mean()**2)
    print(f"T={T:.2f} U={U:.5f} time={time.time()-t0:.1f}s seed={seed}",flush=True)
