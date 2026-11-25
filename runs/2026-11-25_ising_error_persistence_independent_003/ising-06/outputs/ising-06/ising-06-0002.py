
import numpy as np, time
N=16
def make(R, rng):
    return rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
def sweep(S, T, rng, masks, acc):
    for m in masks:
        nn = (np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
        dE = 2*S*nn  # int8 ok: max 8
        # acceptance: dE<=0 always; dE=4,8 with prob exp(-dE/T)
        r = rng.random(S.shape)
        flip = m & ((dE<=0) | (r < acc[np.clip(dE,0,8)]))
        S[flip] *= -1
    return S
rng=np.random.default_rng(12345)
masks=[]
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0), ((ii+jj)%2==1)]
for T in [2.269]:
    acc=np.zeros(9); 
    for d in [4,8]: acc[d]=np.exp(-d/T)
    R=1000
    S=make(R,rng)
    t=time.time()
    for k in range(200):
        sweep(S,T,rng,masks,acc)
    dt=time.time()-t
    print("time per sweep (R=1000):",dt/200)
    ms=[]
    for k in range(2000):
        sweep(S,T,rng,masks,acc)
        if k%10==0:
            m=S.reshape(R,-1).mean(1)
            ms.append(np.mean(m**2))
    ms=np.array(ms)
    print("<m^2> avg over time, first/last halves:", ms[:100].mean(), ms[-100:].mean())
