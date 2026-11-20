
import numpy as np, time, glob
N=16; R=800; burn=3000; meas=25000
def run(T,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    maskA=((ii+jj)%2==0); maskB=~maskA
    def sweep(s):
        for mask in (maskA,maskB):
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE=2*s*nb
            p=np.exp(-dE.clip(min=0)/T)
            acc=(rng.random(s.shape)<p)&mask
            s=np.where(acc,-s,s).astype(np.int8)
        return s
    for _ in range(burn): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R)
    for k in range(meas):
        s=sweep(s)
        m=s.sum(axis=(1,2))/(N*N); m2=m*m
        S2+=m2; S4+=m2*m2
    S2/=meas; S4/=meas
    np.savez(f"/work/T{T}_N{N}.npz",S2=S2,S4=S4,n=meas,seed=seed)
t0=time.time()
run(2.0,1002); run(2.1,1003)
print("sim time",time.time()-t0)
def jk(S2,S4):
    R=len(S2)
    U=1-S4.mean()/(3*S2.mean()**2)
    Ujk=np.empty(R)
    tot2=S2.sum(); tot4=S4.sum()
    for c in range(R):
        a2=(tot2-S2[c])/(R-1); a4=(tot4-S4[c])/(R-1)
        Ujk[c]=1-a4/(3*a2*a2)
    err=np.sqrt((R-1)/R*np.sum((Ujk-Ujk.mean())**2))
    return U,err
for T in [1.9,2.0,2.1]:
    d=np.load(f"/work/T{T}_N{N}.npz")
    U,e=jk(d['S2'],d['S4'])
    print(T,"U=%.5f err=%.5f"%(U,e))
