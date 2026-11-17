import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
code = r'''
import numpy as np
N=16
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
MASKS=[((ii+jj)%2==0),((ii+jj)%2==1)]
def init(R,rng,T):
    if T<2.269:
        return np.ones((R,N,N),dtype=np.int8)
    return np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
def sweep(s,T,rng):
    for par in (0,1):
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=(2*s*nn).astype(np.float64)
        p=np.exp(-dE/T)
        acc=(rng.random(s.shape)<p)&MASKS[par]
        s=np.where(acc,-s,s).astype(np.int8)
    return s
def run(T,R,nequil,nmeas,seed,every=1,trace=False):
    rng=np.random.default_rng(seed)
    s=init(R,rng,T)
    for k in range(nequil):
        s=sweep(s,T,rng)
    S2=np.zeros(R); S4=np.zeros(R); S1=np.zeros(R); cnt=0
    tr=[]
    for k in range(nmeas):
        s=sweep(s,T,rng)
        m=s.reshape(R,-1).mean(axis=1)
        m2=m*m
        S2+=m2; S4+=m2*m2; S1+=np.abs(m); cnt+=1
        if trace: tr.append(np.abs(m).mean())
    return S2/cnt,S4/cnt,S1/cnt,(np.array(tr) if trace else None)
'''
open('/work/ising.py','w').write(code)
import sys; sys.path.insert(0,'/work')
import importlib, ising
importlib.reload(ising)
t0=time.time()
for T in [2.269, 1.5]:
    S2,S4,S1,tr=ising.run(T,500,1000,1000,seed=1,trace=True)
    U_r=1-S4/(3*S2**2)
    print("T",T,"time",time.time()-t0)
    print(" U pooled", 1-S4.mean()/(3*S2.mean()**2), " std of per-replica avg-based U", U_r.std(), "mean |m|", S1.mean())
    # autocorrelation of |m| trace averaged over replicas: crude
    print(" trace means over blocks", [round(x,4) for x in tr.reshape(10,-1).mean(1)])
