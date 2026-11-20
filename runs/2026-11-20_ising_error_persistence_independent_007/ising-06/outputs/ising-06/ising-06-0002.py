
import numpy as np, time, os
N=16; R=1000
Ts=np.array([1.9,2.0,2.1,2.2,2.3,2.4])
SEED=12345
EQ=2000
CH=6000
fn='/work/state.npz'
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
if os.path.exists(fn):
    d=np.load(fn)
    s=d['s']; done=int(d['done']); S2=d['S2']; S4=d['S4']; cnt=d['cnt']; rng=np.random.default_rng([SEED,done])
else:
    rng=np.random.default_rng(SEED)
    s=rng.choice([-1,1],size=(len(Ts),R,N,N)).astype(np.int8)
    done=0; S2=np.zeros((len(Ts),R)); S4=np.zeros((len(Ts),R)); cnt=0
Tc=np.repeat(Ts,R).reshape(len(Ts),R,1,1)
s=s.reshape(len(Ts)*R,N,N)  # flatten
Tb=np.repeat(Ts,R).reshape(-1,1,1)
S2=S2.reshape(-1); S4=S4.reshape(-1)
t0=time.time()
target=done+CH
while done<target and time.time()-t0<300:
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=(2*s*nb).astype(np.float64)
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.clip(dE,0,None)/Tb))
        s=np.where(acc&m,-s,s).astype(np.int8)
    done+=1
    if done>EQ:
        mag=s.reshape(s.shape[0],-1).mean(axis=1)
        S2+=mag**2; S4+=mag**4; cnt+=1
np.savez(fn,s=s.reshape(len(Ts),R,N,N),done=done,S2=S2.reshape(len(Ts),R),S4=S4.reshape(len(Ts),R),cnt=cnt)
print("done",done,"cnt",cnt,"time",time.time()-t0)
