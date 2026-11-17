import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def run(T, R, equil, meas, seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    a4=np.exp(-4/T); a8=np.exp(-8/T)
    def sweep(s):
        for p in (0,1):
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            sn=s*nn
            u=rng.random((R,N,N))
            pr=np.where(sn<=0,1.0,np.where(sn==2,a4,a8))
            flip=(u<pr)&masks[p]
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(equil):
        s=sweep(s)
    # per-replica accumulators in two halves
    A=np.zeros((2,3,R))  # [half][m2,m4,|m|]
    h=meas//2
    for t in range(meas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1)
        m2=m*m
        k=0 if t<h else 1
        A[k,0]+=m2; A[k,1]+=m2*m2; A[k,2]+=np.abs(m)
    A/= np.array([h, meas-h])[:,None,None]
    return A  # per-replica time averages
def binder(m2,m4):
    return 1-m4.mean()/(3*m2.mean()**2)
def jack(m2,m4):
    R=len(m2); U=binder(m2,m4)
    T2=m2.sum(); T4=m4.sum()
    Uj=np.array([1-((T4-m4[r])/(R-1))/(3*((T2-m2[r])/(R-1))**2) for r in range(R)])
    return U, np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
t0=time.time()
A=run(2.3,1000,3000,10000,seed=101)
print("time",time.time()-t0)
full_m2=(A[0,0]+A[1,0])/2; full_m4=(A[0,1]+A[1,1])/2
print("U full", jack(full_m2,full_m4))
print("U halves", binder(A[0,0],A[0,1]), binder(A[1,0],A[1,1]))
print("jack halves", jack(A[0,0],A[0,1])[1], "<|m|>",full_m2.mean())
