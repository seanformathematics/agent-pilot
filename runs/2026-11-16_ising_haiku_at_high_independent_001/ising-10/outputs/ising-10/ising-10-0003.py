import numpy as np, time
N=16
i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mA=((i+j)%2==0); mB=~mA
def sweep(s,expt,rng):
    for m in (mA,mB):
        nn = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE = 2*s*nn
        acc = rng.random(s.shape,dtype=np.float32) < expt[dE+8]
        s[acc & m] *= -1
def run(T,R,nsweep,seed,burn):
    rng=np.random.default_rng(seed)
    expt=np.array([min(1.0,np.exp(-d/T)) for d in range(-8,9)],dtype=np.float32)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    for k in range(burn): sweep(s,expt,rng)
    m2=[];m4=[]
    for k in range(nsweep):
        sweep(s,expt,rng)
        m=s.reshape(R,-1).mean(1)
        m2.append(m*m); m4.append(m**4)
    return np.array(m2),np.array(m4)  # shape (nsweep,R)
def ac(x):
    x=x-x.mean(); n=len(x)
    f=np.fft.rfft(x,2*n); c=np.fft.irfft(f*np.conj(f))[:n]; c/=c[0]
    return c
t=time.time()
for T in [2.269,1.5]:
    m2,m4=run(T,1000,1500,11,500)
    U=1-m4.mean(0)/(3*m2.mean(0)**2)
    # per-sweep series averaged over replicas
    M2=m2.mean(1); print("T",T,"time",time.time()-t)
    print(" U(all)",1-m4.mean()/(3*m2.mean()**2))
    print(" m2 series autocorr:", [round(float(x),3) for x in ac(M2)[[1,5,10,20,50,100,200]]])
    # tau int
    c=ac(M2); tau=1+2*np.sum(c[1:200]*(c[1:200]>0)); print(" tau_int m2 ~",tau)
    # replica jackknife error for U
    Rn=m2.shape[1]
    Us=[]
    for r in range(0,Rn,1):
        pass
    S2=m2.sum(0); S4=m4.sum(0); n=m2.shape[0]
    tot2=S2.sum(); tot4=S4.sum()
    Uj=np.array([1-((tot4-S4[r])/((Rn-1)*n))/(3*((tot2-S2[r])/((Rn-1)*n))**2) for r in range(Rn)])
    err=np.sqrt((Rn-1)/Rn*np.sum((Uj-Uj.mean())**2))
    print(" U=",1-(tot4/(Rn*n))/(3*(tot2/(Rn*n))**2)," jack err (1000 reps,1500 sweeps)",err)
