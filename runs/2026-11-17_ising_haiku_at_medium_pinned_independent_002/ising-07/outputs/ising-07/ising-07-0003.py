
import numpy as np, time, json, os
os.makedirs('/work', exist_ok=True)
N=16; R=1000

def run(T, seed, n_eq, n_meas, R=R):
    rng=np.random.default_rng(seed)
    # ordered-ish start for T<Tc-ish, random for others: random signs of ordered start for T<2.27 else random
    if T < 2.269:
        s=np.ones((R,N,N),dtype=np.int8)
        sign=rng.choice([-1,1],size=(R,1,1)).astype(np.int8)
        s=s*sign
    else:
        s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    acc=np.array([np.exp(-4*T*0+ -dE/T) for dE in (4,8)])  # placeholder
    pacc={dE: np.exp(-dE/T) for dE in (4,8)}
    def sweep(s):
        for mk in masks:
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE=2*s*nn  # energy change for flipping: 2*s*nn (values -8..8)
            r=rng.random(s.shape)
            # accept if dE<=0 or r<exp(-dE/T)
            pa=np.where(dE<=0,1.0,np.exp(-np.clip(dE,0,None)/T))
            flip=(r<pa)&mk
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(n_eq):
        s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); cnt=0
    for _ in range(n_meas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1)
        m2=m*m
        S2+=m2; S4+=m2*m2; cnt+=1
    return S2/cnt, S4/cnt

def binder_jack(a2,a4):
    R=len(a2)
    def U(x2,x4):
        M2=x2.mean(); M4=x4.mean()
        return 1-M4/(3*M2*M2)
    full=U(a2,a4)
    # jackknife over replicas
    tot2=a2.sum(); tot4=a4.sum()
    jk=np.array([U((tot2-a2[i])/(R-1),(tot4-a4[i])/(R-1)) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return full, err

t0=time.time()
a2,a4=run(2.269,12345,200,200)
print("time for 400 sweeps R=1000:",time.time()-t0)
print(binder_jack(a2,a4))
