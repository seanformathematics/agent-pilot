import numpy as np, time
N=16
def make_masks():
    i,j=np.indices((N,N))
    return ((i+j)%2==0), ((i+j)%2==1)
def run(T,R,nsweep,ntherm,seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ev,od=make_masks()
    # acceptance table indexed by dE in {-8..8} for dE=2*s*nn
    acc=np.zeros(17); 
    for d in range(-8,9): acc[d+8]=min(1.0,np.exp(-d/T))
    def half(s,mask):
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=(2*s*nn).astype(np.int8)
        p=acc[dE+8]
        flip=(rng.random(s.shape)<p)&mask
        s[flip]*=-1
    for _ in range(ntherm):
        half(s,ev); half(s,od)
    m2=[];m4=[]
    for k in range(nsweep):
        half(s,ev); half(s,od)
        M=s.sum(axis=(1,2),dtype=np.int64)/(N*N)
        m2.append(M*M); m4.append(M**4)
    return np.array(m2), np.array(m4)
t=time.time()
m2,m4=run(2.2,1000,100,100,1)
print("time for 200 sweeps R=1000:",time.time()-t)
