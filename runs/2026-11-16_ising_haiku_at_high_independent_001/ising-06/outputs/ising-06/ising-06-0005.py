import numpy as np, time, os
N=16
i,j=np.indices((N,N)); EV=((i+j)%2==0); OD=~EV
def sweep(s,T,rng):
    for mask in (EV,OD):
        nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=2*s*nn
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.maximum(dE,0)/T))
        s[acc&mask]*=-1
def run(T,R,nburn,nmeas,seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int64)
    s[:R//2]=1
    for _ in range(nburn): sweep(s,T,rng)
    S2=np.zeros(R); S4=np.zeros(R); S1=np.zeros(R)
    for k in range(nmeas):
        sweep(s,T,rng)
        m=s.reshape(R,-1).mean(1)
        S2+=m*m; S4+=m**4; S1+=np.abs(m)
    return S2/nmeas, S4/nmeas, S1/nmeas
def U(a,b): return 1-b.mean()/(3*a.mean()**2)
def jack(m2r,m4r):
    R=len(m2r); full=U(m2r,m4r)
    loo=np.array([U(np.delete(m2r,k),np.delete(m4r,k)) for k in range(R)])
    return full,np.sqrt((R-1)/R*np.sum((loo-loo.mean())**2))
Ts=[2.2,2.3,2.4,2.5,2.6,2.7,2.8]
for T in Ts:
    t=time.time()
    m2r,m4r,m1r=run(T,1000,2000,4000,int(T*1000)+11)
    full,err=jack(m2r,m4r)
    h=500
    uo=U(m2r[:h],m4r[:h]); ur=U(m2r[h:],m4r[h:])
    np.savez(f'/work/res/T{T}_N16_b.npz',m2=m2r,m4=m4r,m1=m1r)
    print(T,"U=%.5f err=%.5f"%(full,err),"ordstart %.5f randstart %.5f"%(uo,ur),"time %.1f"%(time.time()-t),flush=True)
