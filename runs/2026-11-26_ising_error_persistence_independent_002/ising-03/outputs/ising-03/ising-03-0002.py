import numpy as np, time
N=16
def run_bench(B,T,nsweep,seed=1):
    rng=np.random.Generator(np.random.PCG64(seed))
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(B,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    # acceptance table indexed by (s*h+4)//2 : values for s*h in {-4,-2,0,2,4}
    sh=np.array([-4,-2,0,2,4])
    acc=np.where(sh<=0,1.0,np.exp(-2.0*sh/T)).astype(np.float32)
    t0=time.time()
    for sw in range(nsweep):
        for p in (0,1):
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            sh_=s*nb
            idx=(sh_+4)//2
            pr=acc[idx]
            r=rng.random((B,N,N),dtype=np.float32)
            flip=(r<pr)&masks[p]
            s=np.where(flip,-s,s).astype(np.int8)
    dt=time.time()-t0
    return dt/nsweep
for B in (200,1000):
    t=run_bench(B,2.27,20)
    print(B, "sec/sweep",t, "replica-sweeps/s", B/t)