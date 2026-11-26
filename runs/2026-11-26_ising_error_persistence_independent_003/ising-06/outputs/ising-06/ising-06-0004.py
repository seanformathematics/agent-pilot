import numpy as np, time
N=16
i,j=np.indices((N,N))
MA=((i+j)%2==0); MB=~MA
def sweep(s,rng,acc):
    for mask in (MA,MB):
        nn = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE = 2*s*nn
        u = rng.random(s.shape,dtype=np.float32)
        flip = (u < acc[dE+8]) & mask
        s[flip] *= -1
def run(T,R,nequil,nmeas,seed,nblk=4):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    acc=np.array([1.0 if d<=0 else np.exp(-d/T) for d in range(-8,9)],dtype=np.float32)
    for _ in range(nequil): sweep(s,rng,acc)
    S2=np.zeros((nblk,R)); S4=np.zeros((nblk,R)); per=nmeas//nblk
    for b in range(nblk):
        for _ in range(per):
            sweep(s,rng,acc)
            m=s.reshape(R,-1).mean(1); m2=m*m
            S2[b]+=m2; S4[b]+=m2*m2
    return S2/per,S4/per
def binder_jk(A2,A4):
    R=len(A2); M2=A2.sum(); M4=A4.sum()
    full=1-(M4/R)/(3*(M2/R)**2)
    a2=(M2-A2)/(R-1); a4=(M4-A4)/(R-1)
    loo=1-a4/(3*a2*a2)
    return full,np.sqrt((R-1)/R*np.sum((loo-loo.mean())**2))
t0=time.time()
for k,T in enumerate([2.2,2.3,2.4]):
    S2,S4=run(T,2000,4000,4000,501+k)
    U,err=binder_jk(S2.mean(0),S4.mean(0))
    print(T,"U=%.5f err=%.5f"%(U,err),"blocks",np.round([binder_jk(S2[b],S4[b])[0] for b in range(4)],5),round(time.time()-t0,1),flush=True)