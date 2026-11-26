
import numpy as np, time, pickle
N=16
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
R=1000
d=pickle.load(open('/work/part1.pkl','rb'))
S=d['S']
rng=np.random.default_rng()
rng.bit_generator.state=d['rng']
Tarr=np.repeat(np.array(Ts),R)
beta=(1.0/Tarr).astype(np.float32).reshape(-1,1,1)
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
blocks2=list(d['b2']);blocks4=list(d['b4'])
for b in range(80):
    s2=np.zeros(len(Tarr));s4=np.zeros(len(Tarr))
    for k in range(100):
        sweep(S)
        m=np.abs(S.reshape(len(Tarr),-1).mean(axis=1))
        s2+=m**2; s4+=m**4
    blocks2.append(s2); blocks4.append(s4)
    if time.time()-t0>500:
        print('stop at',b,flush=True); break
pickle.dump(dict(S=S,rng=rng.bit_generator.state,b2=np.array(blocks2),b4=np.array(blocks4)),open('/work/all.pkl','wb'))
print('done',time.time()-t0,len(blocks2))
