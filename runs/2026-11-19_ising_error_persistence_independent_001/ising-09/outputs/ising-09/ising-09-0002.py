import numpy as np, time
N=16; R=1000
rng=np.random.default_rng(12345)
def run(T,sweeps,R=R):
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    # acceptance table indexed by s*h where h in {-4,-2,0,2,4}
    acc={k:np.exp(-2.0*k/T) for k in (-8,-4,0,4,8)}
    for sw in range(sweeps):
        for m in masks:
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            sh=(s*h)
            p=np.where(sh<=0,1.0,np.where(sh==2,np.exp(-4/T),np.where(sh==4,np.exp(-8/T),0.0)))
            flip=(rng.random((R,N,N))<p)&m
            s=np.where(flip,-s,s).astype(np.int8)
    return s
t=time.time(); s=run(2.269,50); print("time per sweep",(time.time()-t)/50)