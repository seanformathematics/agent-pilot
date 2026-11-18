import numpy as np, time
N=16
ii,jj=np.indices((N,N)); par=(ii+jj)%2
masks=[par==0, par==1]
def run(T,R,nsweep_eq,nmeas,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    xs=np.arange(-4,5)
    tab=np.minimum(1.0,np.exp(-2.0*xs/T))
    def sweep(s):
        for mk in masks:
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            x=(s*h).astype(np.int8)
            p=tab[x+4]
            flip=(rng.random(s.shape)<p)&mk
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nsweep_eq): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R)
    for _ in range(nmeas):
        s=sweep(s)
        m=s.sum(axis=(1,2),dtype=np.int64)/(N*N)
        m2=m*m
        S2+=m2; S4+=m2*m2
    return S2/nmeas, S4/nmeas
t=time.time()
S2,S4=run(2.269,500,50,100,1)
print("time",time.time()-t)
t=time.time()
S2,S4=run(2.269,500,0,200,2)
print("time per 200 sweeps R=500",time.time()-t)
