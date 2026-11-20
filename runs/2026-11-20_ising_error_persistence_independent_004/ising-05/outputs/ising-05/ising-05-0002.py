
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def make_masks():
    i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    return ((i+j)%2==0), ((i+j)%2==1)
MA,MB=make_masks()

def nbsum(s):
    return (np.roll(s,1,axis=1)+np.roll(s,-1,axis=1)+np.roll(s,1,axis=2)+np.roll(s,-1,axis=2))

def run(T,R,nburn,nmeas,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    # acceptance table indexed by s*nb (in {-4,-2,0,2,4}) -> dE=2*s*nb
    acc={}
    for x in [-4,-2,0,2,4]:
        dE=2*x
        acc[x]=1.0 if dE<=0 else np.exp(-dE/T)
    tab=np.zeros(9,dtype=np.float64)  # index x+4
    for x in [-4,-2,0,2,4]: tab[x+4]=acc[x]
    S2=np.zeros(nmeas); S4=np.zeros(nmeas); SM=np.zeros(nmeas)
    tot=nburn+nmeas
    for t in range(tot):
        for M in (MA,MB):
            nb=nbsum(s)
            x=(s.astype(np.int16)*nb.astype(np.int16))
            p=tab[(x+4)]
            r=rng.random((R,N,N))
            flip=(r<p)&M
            s=np.where(flip,-s,s).astype(np.int8)
        if t>=nburn:
            m=s.reshape(R,-1).mean(axis=1)
            k=t-nburn
            S2[k]=np.sum(m*m); S4[k]=np.sum(m**4); SM[k]=np.sum(np.abs(m))
    return S2,S4,SM

t0=time.time()
S2,S4,SM=run(2.269,200,50,50,1)
print('time',time.time()-t0)
