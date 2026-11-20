
import numpy as np, time, os, glob
for f in glob.glob('/work/*.npz'): os.remove(f)
N=16; R=1000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T,rng):
    for par in (0,1):
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int16)
        dE=2*s*nb
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.clip(dE,0,None)/T))
        s[acc&masks[par]]*=-1
    return s
def run(T,seed,neq=3000,nmeas=10000):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    for _ in range(neq): sweep(s,T,rng)
    # per-replica sums: [first half, second half] of m^2, m^4
    S=np.zeros((2,2,R))  # half, (m2,m4), replica
    for k in range(nmeas):
        sweep(s,T,rng)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        h=0 if k<nmeas//2 else 1
        S[h,0]+=m*m; S[h,1]+=m**4
    return S
for idx,T in enumerate([1.9,2.0,2.1]):
    t=time.time(); seed=1000+idx
    S=run(T,seed)
    np.savez(f'/work/sums_T{T}_seed{seed}.npz',S=S,T=T,seed=seed)
    M2=S[:,0].sum()/10000/R; M4=S[:,1].sum()/10000/R
    print(T,seed,'U=',1-M4/(3*M2**2),'time',time.time()-t,flush=True)
print(os.popen('df -h /work').read())
