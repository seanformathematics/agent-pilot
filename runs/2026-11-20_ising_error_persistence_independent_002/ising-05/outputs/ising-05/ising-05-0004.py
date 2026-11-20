
import numpy as np, time
N=16; R=1000; NEQ=3000; NMEAS=20000
I,J=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((I+J)%2==c) for c in (0,1)]
def run(T,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep(s):
        for c in (0,1):
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE=2*s*h
            p=np.exp(-np.clip(dE,0,None)/T)
            acc=(rng.random(s.shape)<p)&masks[c]
            s=np.where(acc,-s,s).astype(np.int8)
        return s
    for _ in range(NEQ): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); cnt=0
    for k in range(NMEAS):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m; S2+=m2; S4+=m2*m2; cnt+=1
        if k==NMEAS//2-1: S2a=S2.copy(); S4a=S4.copy()
    np.savez(f"/work/res_N{N}_T{T}.npz",S2=S2,S4=S4,cnt=cnt,S2a=S2a,S4a=S4a,n_half=NMEAS//2,seed=seed,NEQ=NEQ)
t0=time.time()
for T,seed in [(2.3,1005),(2.4,1006)]:
    run(T,seed); print("done",T,time.time()-t0,flush=True)
