
import numpy as np, time
N=16; R=1000
Ts=np.array([1.9,2.0,2.1,2.2,2.3,2.4]); nT=len(Ts)
SEED=12345
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
d=np.load('/work/state.npz')
s=d['s'].reshape(nT*R,N,N); done=int(d['done'])
rng=np.random.default_rng([SEED,done])
Tb=np.repeat(Ts,R).reshape(-1,1,1)
S2b=np.zeros(nT*R); S4b=np.zeros(nT*R); cntb=0
t0=time.time(); target=done+3000
while done<target and time.time()-t0<400:
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=(2*s*nb).astype(np.float64)
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.clip(dE,0,None)/Tb))
        s=np.where(acc&m,-s,s).astype(np.int8)
    done+=1
    mag=s.reshape(s.shape[0],-1).mean(axis=1)
    S2b+=mag**2; S4b+=mag**4; cntb+=1
np.savez('/work/state2.npz',s=s.reshape(nT,R,N,N),done=done,S2b=S2b.reshape(nT,R),S4b=S4b.reshape(nT,R),cntb=cntb,S2=d['S2'],S4=d['S4'],cnt=d['cnt'])
S2=d['S2']; S4=d['S4']; cnt=int(d['cnt'])
def U_and_err(S2x,S4x,c):
    R=S2x.shape[0]
    tot2=S2x.sum(); tot4=S4x.sum()
    U=1-(tot4/(R*c))/(3*(tot2/(R*c))**2)
    a2=(tot2-S2x)/((R-1)*c); a4=(tot4-S4x)/((R-1)*c)
    Uj=1-a4/(3*a2**2)
    return U, np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
print("done",done,"cntb",cntb,"time",time.time()-t0)
for k,T in enumerate(Ts):
    Uo,eo=U_and_err(S2[k],S4[k],cnt)
    Un,en=U_and_err(S2b.reshape(nT,R)[k],S4b.reshape(nT,R)[k],cntb)
    Ua,ea=U_and_err(S2[k]+S2b.reshape(nT,R)[k],S4[k]+S4b.reshape(nT,R)[k],cnt+cntb)
    print(T, "old %.5f±%.5f  new %.5f±%.5f  all %.5f±%.5f"%(Uo,eo,Un,en,Ua,ea))
