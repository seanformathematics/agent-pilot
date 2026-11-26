
import numpy as np, time, os, pickle
N=16
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
R=1000
SEED=20240601
rng=np.random.default_rng(SEED)
Tarr=np.repeat(np.array(Ts),R)
beta=(1.0/Tarr).astype(np.float32).reshape(-1,1,1)
S=rng.choice(np.array([-1,1],dtype=np.int8),size=(len(Tarr),N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(S):
    for m in masks:
        nn=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.float32)
        dE=2*S*nn
        acc=(dE<=0)|(rng.random(S.shape,dtype=np.float32)<np.exp(-beta*dE))
        S[acc&m]*=-1
    return S
t0=time.time()
nsw=0
for _ in range(2000):
    sweep(S); nsw+=1
print('equil done',time.time()-t0,flush=True)
blocks2=[];blocks4=[]
# measure 4000 sweeps this call = 40 blocks of 100
for b in range(40):
    s2=np.zeros(len(Tarr));s4=np.zeros(len(Tarr))
    for k in range(100):
        sweep(S)
        m=np.abs(S.reshape(len(Tarr),-1).mean(axis=1))
        s2+=m**2; s4+=m**4
    blocks2.append(s2); blocks4.append(s4)
    if time.time()-t0>540: 
        print('stopping early at block',b,flush=True); break
pickle.dump(dict(S=S,rng=rng.bit_generator.state,b2=np.array(blocks2),b4=np.array(blocks4)),open('/work/part1.pkl','wb'))
print('done',time.time()-t0,len(blocks2))
