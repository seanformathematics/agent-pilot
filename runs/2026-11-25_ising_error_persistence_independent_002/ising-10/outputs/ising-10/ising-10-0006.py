
import numpy as np, time
N=16; R=1000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def run(T, seed, n_eq=3000, n_meas=20000, blk=1000):
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep(S):
        for p in (0,1):
            nn=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
            dE=2*S*nn
            r=rng.random(S.shape)
            flip=((dE<=0)|(r<np.exp(-dE/T)))&masks[p]
            S=np.where(flip,-S,S)
        return S
    for k in range(n_eq): S=sweep(S)
    m2s=np.zeros(R); m4s=np.zeros(R); m2b=[]; b2=np.zeros(R)
    for k in range(n_meas):
        S=sweep(S)
        m=S.sum(axis=(1,2))/(N*N); m2=m*m
        m2s+=m2; m4s+=m2*m2; b2+=m2
        if (k+1)%blk==0:
            m2b.append(b2/blk); b2=np.zeros(R)
    return m2s/n_meas, m4s/n_meas, np.array(m2b)
for T,seed in [(2.3,105),(2.4,106)]:
    t=time.time()
    a2,a4,blocks=run(T,seed)
    np.savez(f'/work/res_T{T}.npz',a2=a2,a4=a4,blocks=blocks)
    print(T,"time",round(time.time()-t),flush=True)

def jk(a2,a4):
    U=1-a4.mean()/(3*a2.mean()**2)
    n=len(a2); s2=a2.sum(); s4=a4.sum()
    Uj=np.array([1-((s4-a4[i])/(n-1))/(3*((s2-a2[i])/(n-1))**2) for i in range(n)])
    return U, np.sqrt((n-1)/n*np.sum((Uj-Uj.mean())**2))
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/res_T{T}.npz')
    U,e=jk(d['a2'],d['a4'])
    print(T, "U=%.6f err=%.6f  nrep=%d  meas_sweeps=%d"%(U,e,len(d['a2']),20000))
