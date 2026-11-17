
import numpy as np, time, json, os
os.makedirs('/work',exist_ok=True)
N=16
i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mA=((i+j)%2==0); mB=~mA
def run_series(R,T,nburn,nmeas,seed):
    rng=np.random.default_rng(seed)
    s = rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    p=np.array([1.0 if 2*x<=0 else np.exp(-2*x/T) for x in range(-4,5)])
    def sweep(s):
        for mask in (mA,mB):
            nn = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            x = (s*nn).astype(np.int8)+4
            acc = rng.random((R,N,N)) < p[x]
            s = np.where(acc & mask,-s,s).astype(np.int8)
        return s
    for _ in range(nburn): s=sweep(s)
    m2s=np.zeros(R); m4s=np.zeros(R); absm=np.zeros(R)
    series_m2=[]; series_m4=[]
    for k in range(nmeas):
        s=sweep(s)
        m=s.mean(axis=(1,2)); m2=m*m
        m2s+=m2; m4s+=m2*m2
    return m2s/nmeas, m4s/nmeas
def jack_U(a2,a4,nb=50):
    R=len(a2)
    idx=np.array_split(np.arange(R),nb)
    U=1-a4.mean()/(3*a2.mean()**2)
    Uj=[]
    for ix in idx:
        mask=np.ones(R,bool); mask[ix]=False
        Uj.append(1-a4[mask].mean()/(3*a2[mask].mean()**2))
    Uj=np.array(Uj)
    err=np.sqrt((nb-1)/nb*np.sum((Uj-Uj.mean())**2))
    return U,err
temps=[1.5,1.6,1.7,1.8,1.9,2.0,2.1]
res={}
for T in temps:
    t0=time.time()
    a2,a4=run_series(1000,T,1000,4000,seed=int(round(T*1000))+7)
    U,e=jack_U(a2,a4)
    res[T]=dict(U=U,err=e,n=1000)
    print(T,U,e,time.time()-t0,flush=True)
json.dump(res,open('/work/partA.json','w'))
