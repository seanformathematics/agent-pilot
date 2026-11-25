
import numpy as np, time
N=16; R=2000; M=20000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def run(T,seed):
    rng=np.random.default_rng(seed)
    beta=1/T
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep(s):
        for p in (0,1):
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.float32)
            x=(s*nn)
            acc=np.exp(-2*beta*np.maximum(x,0))
            u=rng.random(s.shape,dtype=np.float32)
            flip=(u<acc)&masks[p]
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(4000): s=sweep(s)
    S2=np.zeros((2,R)); S4=np.zeros((2,R))
    for k in range(M):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        h=0 if k<M//2 else 1
        S2[h]+=m*m; S4[h]+=m**4
    np.savez(f"/work/res_T{T}_N{N}.npz",S2=S2,S4=S4,M=M,R=R,seed=seed)
t0=time.time()
run(2.0,1002); print("2.0 done",time.time()-t0)
run(2.1,1003); print("2.1 done",time.time()-t0)
