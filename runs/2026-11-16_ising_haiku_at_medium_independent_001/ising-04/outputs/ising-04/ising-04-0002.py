
import numpy as np, json, time
exec(open('/dev/stdin').read()) if False else None
N=16
def run(T, R=1000, sweeps=8000, burn=1000, seed=1):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    acc={4:np.exp(-4/T),8:np.exp(-8/T)}
    s1=np.zeros(R); s2=np.zeros(R); cnt=0
    for sw in range(sweeps):
        for m in masks:
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nn
            p=np.ones(dE.shape)
            p=np.where(dE==4,acc[4],p); p=np.where(dE==8,acc[8],p)
            u=rng.random(s.shape)
            flip=m[None]&((dE<=0)|(u<p))
            s=np.where(flip,-s,s).astype(np.int8)
        if sw>=burn:
            mag=s.reshape(R,-1).mean(axis=1)
            m2=mag*mag; s1+=m2; s2+=m2*m2; cnt+=1
    return s1/cnt, s2/cnt

def U_and_err(a2,a4):
    R=len(a2)
    U=1-a4.mean()/(3*a2.mean()**2)
    # jackknife over replicas
    tot2=a2.sum(); tot4=a4.sum()
    Uj=np.empty(R)
    for k in range(R):
        m2=(tot2-a2[k])/(R-1); m4=(tot4-a4[k])/(R-1)
        Uj[k]=1-m4/(3*m2*m2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err

temps=[1.5,1.6,1.7,1.8,1.9,2.0,2.1]
out={}
t0=time.time()
for i,T in enumerate(temps):
    a2,a4=run(T,seed=1000+i)
    U,e=U_and_err(a2,a4)
    out[T]=(U,e)
    print(T,U,e,time.time()-t0,flush=True)
    np.savez(f'/work/A_{T}.npz',a2=a2,a4=a4)
