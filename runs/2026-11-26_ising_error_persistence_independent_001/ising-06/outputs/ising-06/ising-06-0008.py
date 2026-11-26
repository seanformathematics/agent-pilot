import numpy as np, time
from scipy.special import ellipk
def onsager_e(T):
    K=1/T; k=2*np.sinh(2*K)/np.cosh(2*K)**2
    return -(1/np.tanh(2*K))*(1+(2/np.pi)*(2*np.tanh(2*K)**2-1)*ellipk(k*k))
def run(N,T,R=400,nsw=600,seed=1):
    i,j=np.indices((N,N)); masks=[((i+j)%2==p) for p in (0,1)]
    rng=np.random.default_rng(seed); beta=np.float32(1/T)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    es=[]
    for k in range(nsw):
        for mk in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nb
            acc=np.exp(-beta*np.maximum(dE,0).astype(np.float32))
            r=rng.random(s.shape,dtype=np.float32)
            s[(r<acc)&mk]*=-1
        if k>nsw//2:
            es.append(-(np.roll(s,1,1)*s+np.roll(s,1,2)*s).reshape(R,-1).mean(1).mean())
    return np.mean(es)
for T in [2.4,3.0]:
    print("T",T,"onsager",round(onsager_e(T),4))
    for N in [8,16,32]:
        t0=time.time()
        print("  N",N,round(run(N,T,R=200 if N==32 else 400,nsw=400),4), round(time.time()-t0,1),flush=True)