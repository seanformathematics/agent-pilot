import numpy as np, time, os
N=16
i,j=np.indices((N,N)); EV=((i+j)%2==0); OD=~EV
def sweep(s,T,rng):
    for mask in (EV,OD):
        nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=2*s*nn
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.maximum(dE,0)/T))
        s[acc&mask]*=-1
def run(T,R,nburn,nmeas,seed,start):
    rng=np.random.default_rng(seed)
    if start=='rand': s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int64)
    else: s=np.ones((R,N,N),dtype=np.int64)
    for _ in range(nburn): sweep(s,T,rng)
    S2=np.zeros(R); S4=np.zeros(R); S1=np.zeros(R)
    for k in range(nmeas):
        sweep(s,T,rng)
        m=s.reshape(R,-1).mean(1)
        S2+=m*m; S4+=m**4; S1+=np.abs(m)
    return S2/nmeas, S4/nmeas, S1/nmeas
def jack(m2r,m4r):
    R=len(m2r)
    def U(a,b): return 1-b.mean()/(3*a.mean()**2)
    full=U(m2r,m4r)
    loo=np.array([U(np.delete(m2r,k),np.delete(m4r,k)) for k in range(R)])
    err=np.sqrt((R-1)/R*np.sum((loo-loo.mean())**2))
    return full,err
os.makedirs('/work/res',exist_ok=True)
for T in [2.3,2.2]:
    t=time.time()
    m2r,m4r,m1r=run(T,1000,2000,8000,int(T*1000)+7,'ord' if T<2.27 else 'rand')
    full,err=jack(m2r,m4r)
    np.savez(f'/work/res/T{T}_N16.npz',m2=m2r,m4=m4r,m1=m1r)
    print(T,"U=%.5f err=%.5f"%(full,err),"time %.1f"%(time.time()-t), "<|m|>=%.4f"%m1r.mean())
