
import numpy as np, time
N=16;R=200
mask=(np.add.outer(np.arange(N),np.arange(N))%2==0)
def run(T,seed,neq,nm):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    m2=[];m4=[]
    for it in range(neq+nm):
        for p in (mask,~mask):
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            pu=1/(1+np.exp(-2*nb/T))
            new=np.where(rng.random(s.shape)<pu,1,-1).astype(np.int8)
            s=np.where(p,new,s)
        if it>=neq:
            m=s.sum((1,2))/N**2
            m2.append(m**2);m4.append(m**4)
    return np.array(m2),np.array(m4)
res={}
for i,T in enumerate([1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]):
    t=time.time()
    a,b=run(T,100+i,500,4000)
    # per-replica U jackknife-free: replica-level bootstrap
    rng=np.random.default_rng(1)
    A=a.mean(0);B=b.mean(0)
    U=1-B.mean()/(3*A.mean()**2)
    bs=[]
    for _ in range(300):
        k=rng.integers(0,R,R);bs.append(1-B[k].mean()/(3*A[k].mean()**2))
    print(T,U,np.std(bs),time.time()-t,flush=True)
