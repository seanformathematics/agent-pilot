
import numpy as np, time, json
N=16; R=400
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T,rng):
    for mk in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nb
        acc=np.exp(-dE/T)
        u=rng.random(s.shape,dtype=np.float32)
        flip=(u<acc)&mk
        s[flip]*=-1
def jack_U(m2,m4):
    # m2,m4: per-replica time averages, shape (R,)
    R=len(m2)
    S2=m2.sum(); S4=m4.sum()
    U=1-(S4/R)/(3*(S2/R)**2)
    Us=np.empty(R)
    for k in range(R):
        a2=(S2-m2[k])/(R-1); a4=(S4-m4[k])/(R-1)
        Us[k]=1-a4/(3*a2**2)
    Uerr=np.sqrt((R-1)/R*np.sum((Us-Us.mean())**2))
    return U,Uerr
res={}
seed=12345
for T in Ts:
    t0=time.time()
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    for _ in range(1000): sweep(s,T,rng)
    m2s=np.zeros(R); m4s=np.zeros(R)
    nm=6000
    series=[]
    for _ in range(nm):
        sweep(s,T,rng)
        m=s.reshape(R,-1).mean(1)
        m2s+=m**2; m4s+=m**4
    m2s/=nm; m4s/=nm
    U,Ue=jack_U(m2s,m4s)
    res[T]=dict(U=U,Uerr=Ue,seed=seed)
    print(T,round(U,5),round(Ue,5),round(time.time()-t0,1),flush=True)
    seed+=1
json.dump(res,open('/work/run1.json','w'))
