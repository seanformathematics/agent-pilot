
import numpy as np, time, sys
N=16;R=256
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
mask=((np.indices((N,N)).sum(0))%2==0)
masks=[mask,~mask]
def run(T,seed,neq,nmeas,every):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    acc=np.exp(-2*np.array([-8,-4,0,4,8])/T)  # index by (s*nb+8)/4... dE=2*s*nb
    def sweep():
        for mk in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb
            p=np.exp(-dE/T)
            flip=(rng.random(s.shape)<p)&mk
            s[flip]*=-1
    for _ in range(neq): sweep()
    m2=np.zeros(R);m4=np.zeros(R);n=0
    for i in range(nmeas):
        for _ in range(every): sweep()
        m=s.sum((1,2))/N**2
        m2+=m**2;m4+=m**4;n+=1
    return m2/n,m4/n,n
t=time.time()
a=run(2.3,1,200,100,5);print(time.time()-t)
