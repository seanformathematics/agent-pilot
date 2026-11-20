
import numpy as np, time
N=16; R=2000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
def sweep(s,T,rng):
    for c in (0,1):
        nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=2*s*nb
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.clip(dE,0,None)/T))
        s=np.where(acc&masks[c],-s,s).astype(np.int8)
    return s
def run(T,seed,neq=3000,nmeas=8000):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    for _ in range(neq): s=sweep(s,T,rng)
    S2=np.zeros(R); S4=np.zeros(R)
    for k in range(nmeas):
        s=sweep(s,T,rng)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m; S2+=m2; S4+=m2*m2
    return S2/nmeas, S4/nmeas
def U(x2,x4): return 1-x4.mean()/(3*x2.mean()**2)
def ujack(a2,a4):
    n=len(a2); jk=np.empty(n); tot2=a2.sum(); tot4=a4.sum()
    for r in range(n): jk[r]=U((tot2-a2[r])/(n-1),(tot4-a4[r])/(n-1))
    return U(a2,a4), np.sqrt((n-1)/n*np.sum((jk-jk.mean())**2))
t=time.time()
a2,a4=run(2.4,106)
u,e=ujack(a2,a4); print(2.4,u,e,a2.mean(),time.time()-t,flush=True)
np.savez("/work/raw_T2.4.npz",a2=a2,a4=a4)
# burn-in check at T=2.3
a2b,a4b=run(2.3,205,neq=300,nmeas=3000)
ub,eb=ujack(a2b,a4b)
print("short burn-in T=2.3:",ub,eb,flush=True)
