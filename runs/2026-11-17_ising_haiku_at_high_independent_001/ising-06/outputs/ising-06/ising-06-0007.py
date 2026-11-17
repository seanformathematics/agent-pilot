import numpy as np, time, os, json
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
    A=np.zeros((2,3,R))
    h=meas//2
    for t in range(meas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1)
        m2=m*m
        k=0 if t<h else 1
        A[k,0]+=m2; A[k,1]+=m2*m2; A[k,2]+=np.abs(m)
    A/= np.array([h, meas-h])[:,None,None]
    return A
def binder(m2,m4):
    return 1-m4.mean()/(3*m2.mean()**2)
def jack(m2,m4):
    R=len(m2); U=binder(m2,m4)
    T2=m2.sum(); T4=m4.sum()
    Uj=(1-((T4-m4)/(R-1))/(3*((T2-m2)/(R-1))**2))
    return U, np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
temps=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
out={}
t0=time.time()
for i,T in enumerate(temps):
    A=run(T,500,3000,4000,seed=1000+i)
    m2=(A[0,0]+A[1,0])/2; m4=(A[0,1]+A[1,1])/2
    U,e=jack(m2,m4)
    Uh1=binder(A[0,0],A[0,1]); Uh2=binder(A[1,0],A[1,1])
    out[T]=dict(U=float(U),err=float(e),Uh1=float(Uh1),Uh2=float(Uh2),m=float(np.mean(A[:,2])))
    print(T, round(U,5), round(e,6), "halves", round(Uh1,5), round(Uh2,5), "<|m|>",round(np.mean(A[:,2]),4), "t",round(time.time()-t0,1), flush=True)
    json.dump(out,open('/work/results_prod1.json','w'))
