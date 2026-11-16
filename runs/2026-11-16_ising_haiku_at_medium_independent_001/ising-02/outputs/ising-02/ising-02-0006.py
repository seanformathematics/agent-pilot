
import numpy as np, json, time
N=16; R=500; NEQ=1000; NMEAS=3000; NGRP=20
Ts=[2.2,2.3,2.4,2.5,2.6,2.7,2.8]
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def run(T,seed):
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep():
        for m in masks:
            nb=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
            dE=2*S*nb
            acc=np.exp(-np.clip(dE,0,None)/T)
            flip=(rng.random(S.shape)<acc)&m
            S[flip]*=-1
    for _ in range(NEQ): sweep()
    a2=np.zeros(R); a4=np.zeros(R)
    for t in range(NMEAS):
        sweep()
        mm=S.reshape(R,-1).mean(axis=1,dtype=np.float64)
        a2+=mm**2; a4+=mm**4
    a2/=NMEAS; a4/=NMEAS
    groups=np.array_split(np.arange(R),NGRP)
    U=1-a4.mean()/(3*a2.mean()**2)
    Uj=[]
    for g in groups:
        keep=np.ones(R,bool); keep[g]=False
        Uj.append(1-a4[keep].mean()/(3*a2[keep].mean()**2))
    Uj=np.array(Uj)
    err=np.sqrt((NGRP-1)/NGRP*np.sum((Uj-Uj.mean())**2))
    return dict(T=T,U=float(U),U_err=float(err))
t0=time.time(); out=[]
for k,T in enumerate(Ts):
    r=run(T,3000+k); out.append(r); print(r, round(time.time()-t0,1), flush=True)
json.dump(out,open('/work/resB.json','w'))
