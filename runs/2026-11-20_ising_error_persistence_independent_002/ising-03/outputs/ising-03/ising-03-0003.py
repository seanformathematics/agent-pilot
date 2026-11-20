import numpy as np, time, json
N=16
def run(T, R, neq, nmeas, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p)[None] for p in (0,1)]
    pA=np.exp(-4/T); pB=np.exp(-8/T)
    def sweep():
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            sn=s*nb
            p=np.where(sn<=0,1.0,np.where(sn==2,pA,pB))
            r=rng.random(s.shape)
            flip=m & (r<p)
            s[flip]*=-1
    for _ in range(neq): sweep()
    S2=np.zeros(R); S4=np.zeros(R); Sabs=np.zeros(R)
    blocks=[]
    for k in range(nmeas):
        sweep()
        m=s.reshape(R,-1).mean(1)
        S2+=m**2; S4+=m**4; Sabs+=np.abs(m)
    return S2/nmeas,S4/nmeas,Sabs/nmeas
def U_and_err(a2,a4):
    R=len(a2)
    U=1-a4.mean()/(3*a2.mean()**2)
    idx=np.arange(R)
    Uj=np.empty(R)
    tot2=a2.sum(); tot4=a4.sum()
    for i in range(R):
        m2=(tot2-a2[i])/(R-1); m4=(tot4-a4[i])/(R-1)
        Uj[i]=1-m4/(3*m2**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
T=2.2
t0=time.time()
a2,a4,aa=run(T,2000,2000,4000,seed=101)
print("time",time.time()-t0)
U,err=U_and_err(a2,a4)
print("T",T,"U",U,"err",err)
np.savez(f"/work/T{T}_s101.npz",a2=a2,a4=a4,aa=aa)
