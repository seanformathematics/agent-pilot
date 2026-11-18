
import numpy as np, time
N=16; R=2000; NEQ=1500; NMEAS=4000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def run(T,seed,R=R,NEQ=NEQ,NMEAS=NMEAS):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ptab=np.array([min(1.0,np.exp(-2*(k-4)/T)) for k in range(9)],dtype=np.float32)
    def sweep(s):
        for m in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            p=ptab[s*nb+4]
            u=rng.random(s.shape,dtype=np.float32)
            s=np.where((u<p)&m,-s,s)
        return s
    for _ in range(NEQ): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R)
    for k in range(NMEAS):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m; S2+=m2; S4+=m2*m2
    return S2/NMEAS,S4/NMEAS
for T in [2.0,2.1]:
    m2r,m4r=run(T,seed=7)
    Ur=1-m4r/(3*m2r**2)
    print('T',T,'U pooled',1-m4r.mean()/(3*m2r.mean()**2),'per-replica U std',Ur.std(ddof=1),'std/sqrt(R)',Ur.std(ddof=1)/np.sqrt(R))
    print('  m2 mean',m2r.mean(),'std',m2r.std(ddof=1))
