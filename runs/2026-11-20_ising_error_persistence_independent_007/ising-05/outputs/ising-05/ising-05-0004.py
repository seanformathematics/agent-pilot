import numpy as np, time, sys
L=16; N=L*L
idx=np.arange(N).reshape(L,L)
up=np.roll(idx,-1,axis=0); dn=np.roll(idx,1,axis=0); rt=np.roll(idx,-1,axis=1); lf=np.roll(idx,1,axis=1)
nbr=np.stack([up.ravel(),dn.ravel(),rt.ravel(),lf.ravel()],axis=1)
ii,jj=np.meshgrid(np.arange(L),np.arange(L),indexing='ij')
black=((ii+jj)%2==0).ravel(); white=~black
bidx=np.where(black)[0]; widx=np.where(white)[0]
def sweep(S,beta,rng):
    for sites in (bidx,widx):
        nn=(S[:,nbr[sites,0]].astype(np.int16)+S[:,nbr[sites,1]]+S[:,nbr[sites,2]]+S[:,nbr[sites,3]])
        s=S[:,sites].astype(np.int16)
        x=s*nn
        p=np.exp(-2*beta*np.clip(x,0,None)).astype(np.float32)
        r=rng.random(p.shape,dtype=np.float32)
        S[:,sites]=np.where(r<p,-s,s).astype(np.int8)
T=float(sys.argv[1]) if len(sys.argv)>1 else 1.9
TT=1.9; seed=int(12345+round(TT*100)); 
def run(T,seed,R=1000,burn=2000,nmeas=20000,blk=50):
    rng=np.random.default_rng(seed)
    beta=1.0/T
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N))
    t0=time.time()
    for _ in range(burn): sweep(S,beta,rng)
    nb=nmeas//blk
    m2b=np.zeros((R,nb)); m4b=np.zeros((R,nb)); absm=np.zeros(nmeas)
    for k in range(nmeas):
        sweep(S,beta,rng)
        m=S.mean(axis=1,dtype=np.float64)
        m2=m*m
        b=k//blk
        m2b[:,b]+=m2; m4b[:,b]+=m2*m2
        absm[k]=np.abs(m).mean()
    m2b/=blk; m4b/=blk
    np.savez(f'/work/T{T:.2f}_s{seed}.npz',m2b=m2b,m4b=m4b,absm=absm,seed=seed)
    print('T',T,'time',time.time()-t0)
    return m2b,m4b,absm
m2b,m4b,absm=run(T,seed+int(T*1000)%997)
M2=m2b.mean(); M4=m4b.mean()
U=1-M4/(3*M2**2)
print('U all',U)
# halves check
h=m2b.shape[1]//2
for a,b in [(slice(0,h),slice(h,None))]:
    print('U first half',1-m4b[:,a].mean()/(3*m2b[:,a].mean()**2),'second half',1-m4b[:,b].mean()/(3*m2b[:,b].mean()**2))
print('absm per 1000 sweeps', [round(absm[i:i+1000].mean(),4) for i in range(0,20000,1000)])
