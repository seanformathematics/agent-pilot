import numpy as np
N=16
def sweep_factory(T,R,rng):
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    a4=np.exp(-4/T); a8=np.exp(-8/T)
    def sweep(s):
        for p in (0,1):
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            sn=s*nn
            u=rng.random((R,N,N))
            pr=np.where(sn<=0,1.0,np.where(sn==2,a4,a8))
            flip=(u<pr)&masks[p]
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    return sweep
rng=np.random.default_rng(5)
for T in [10.0,2.3,1.5]:
    R=200
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    sw=sweep_factory(T,R,rng)
    for _ in range(1000): s=sw(s)
    m=s.reshape(R,-1).mean(1)
    E=-(s*np.roll(s,1,1)+s*np.roll(s,1,2)).reshape(R,-1).sum(1)/N/N
    print(T, "<|m|>",np.abs(m).mean(), "<m2>",(m*m).mean(), "E/N",E.mean())
