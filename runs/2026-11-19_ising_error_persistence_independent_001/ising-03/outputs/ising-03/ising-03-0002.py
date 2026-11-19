
import numpy as np, time, json
N=16
temps=[1.9,2.0,2.1,2.2,2.3,2.4]
def run(T, R, nequil, nmeas, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    # acceptance table for dE = 2*s*h, s*h in {-4..4}
    acc={}
    for d in (4,8):
        acc[d]=np.exp(-2*d/T) if False else np.exp(-d/T)
    def sweep():
        for m in masks:
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            sh=(s*h)  # in {-4,-2,0,2,4}
            dE=2*sh  # energy change when flipping s
            p=np.where(dE<=0,1.0,np.exp(-dE/T))
            r=rng.random(s.shape)
            flip=(r<p)&m
            s[flip]*=-1
    for _ in range(nequil): sweep()
    S2=np.zeros(R); S4=np.zeros(R); cnt=0
    for _ in range(nmeas):
        sweep()
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m
        S2+=m2; S4+=m2*m2; cnt+=1
    return S2/cnt, S4/cnt
def binder_jk(a2,a4):
    R=len(a2)
    def U(idx):
        x2=a2[idx].mean(); x4=a4[idx].mean()
        return 1-x4/(3*x2*x2)
    full=U(np.arange(R))
    jk=np.array([U(np.delete(np.arange(R),i)) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return full,err
t0=time.time()
res={}
for k,T in enumerate(temps):
    a2,a4=run(T,R=200,nequil=300,nmeas=300,seed=1000+k)
    u,e=binder_jk(a2,a4)
    res[T]=(u,e)
    print(T,round(u,5),round(e,5),round(time.time()-t0,1),flush=True)
