
import numpy as np, time, json, os
os.makedirs('/work',exist_ok=True)
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
    m2=np.zeros((nmeas,R)); m4=np.zeros((nmeas,R))
    for k in range(nmeas):
        sweep(s,T,rng)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2[k]=m*m; m4[k]=m**4
    return m2,m4
if __name__=="__main__":
    import sys
    Ts=[float(x) for x in sys.argv[1].split(',')]
    seedbase=int(sys.argv[2])
    for idx,T in enumerate(Ts):
        t=time.time()
        seed=seedbase+idx
        m2,m4=run(T,seed)
        np.savez(f'/work/run_T{T}_seed{seed}.npz',m2=m2,m4=m4,T=T,seed=seed)
        M2=m2.mean(); M4=m4.mean()
        print(T,seed,'U=',1-M4/(3*M2**2),'time',time.time()-t,flush=True)
