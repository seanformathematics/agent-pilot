import numpy as np, time
L=16; N=L*L
idx=np.arange(N).reshape(L,L)
up=np.roll(idx,-1,axis=0); dn=np.roll(idx,1,axis=0); rt=np.roll(idx,-1,axis=1); lf=np.roll(idx,1,axis=1)
nbr=np.stack([up.ravel(),dn.ravel(),rt.ravel(),lf.ravel()],axis=1)
ii,jj=np.meshgrid(np.arange(L),np.arange(L),indexing='ij')
black=((ii+jj)%2==0).ravel(); white=~black
bidx=np.where(black)[0]; widx=np.where(white)[0]
def make(R,rng):
    return rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N))
def sweep(S,T,rng,sub):
    beta=1.0/T
    # acceptance table indexed by s*nn in {-4,-2,0,2,4} -> p = exp(-2*beta*s*nn) when s*nn>0 else 1
    for sites in (bidx,widx):
        nn=(S[:,nbr[sites,0]].astype(np.int16)+S[:,nbr[sites,1]]+S[:,nbr[sites,2]]+S[:,nbr[sites,3]])
        s=S[:,sites].astype(np.int16)
        x=s*nn  # in {-4..4}
        p=np.exp(-2*beta*np.clip(x,0,None)).astype(np.float32)
        r=rng.random(p.shape,dtype=np.float32)
        flip=r<p
        newv=np.where(flip,-s,s).astype(np.int8)
        S[:,sites]=newv
    return S
rng=np.random.default_rng(1)
S=make(1000,rng)
t=time.time()
for _ in range(100): sweep(S,2.2,rng,None)
dt=time.time()-t
print('per sweep (1000 replicas) s:',dt/100)
