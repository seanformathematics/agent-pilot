
import numpy as np, time, sys, os
os.makedirs('/work',exist_ok=True)
N=16; R=1000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def sweep(s,beta,rng):
    for p in (0,1):
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*h
        acc=np.exp(-beta*np.clip(dE,0,None))
        flip=(rng.random(s.shape)<acc)&masks[p]
        s[flip]*=-1
    return s
def run(T,seed,nburn,nmeas):
    rng=np.random.default_rng(seed)
    beta=1/T
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    s=s.astype(np.int8)
    for k in range(nburn): sweep(s,beta,rng)
    m2s=np.zeros(R); m4s=np.zeros(R); absm=np.zeros(R)
    trace=[]
    for k in range(nmeas):
        sweep(s,beta,rng)
        m=s.reshape(R,-1).mean(axis=1)
        m2=m*m
        m2s+=m2; m4s+=m2*m2; absm+=np.abs(m)
        if k%1000==0: trace.append(float(np.abs(m).mean()))
    np.savez(f'/work/acc_T{T}_s{seed}.npz',m2=m2s,m4=m4s,absm=absm,nmeas=nmeas)
    U=1-(m4s/nmeas)/(3*(m2s/nmeas)**2)
    return U, trace
T=float(sys.argv[1]) if len(sys.argv)>1 else 1.9
for T,seed in [(1.9,101),(2.0,102)]:
    t=time.time()
    U,tr=run(T,seed,2000,15000)
    print(T, "U mean over replicas", U.mean(), "sem", U.std(ddof=1)/np.sqrt(R), "time",time.time()-t)
    print("trace |m| every 1000 sweeps", np.round(tr,3))
