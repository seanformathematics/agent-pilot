import numpy as np, time, os
N=16; R=400
def run(T, seed, nequil, nmeas, blk=200):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    k=np.arange(-4,5)
    accT=np.minimum(1.0,np.exp(-2*k/T))
    def sweep(s):
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            idx=(s*nb).astype(np.int16)+4
            flip=(rng.random(s.shape)<accT[idx])&m
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nequil): s=sweep(s)
    nb_=nmeas//blk
    B2=np.zeros((nb_,R)); B4=np.zeros((nb_,R)); BA=np.zeros((nb_,R))
    for t in range(nmeas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1)
        b=t//blk
        B2[b]+=m*m/blk; B4[b]+=m**4/blk; BA[b]+=np.abs(m)/blk
    return B2,B4,BA
def jack(B2,B4):
    Rr=B2.shape[1]
    S2=B2.sum(0); S4=B4.sum(0); n=B2.shape[0]
    tot2=S2.sum(); tot4=S4.sum()
    Uf=1-(tot4/(n*Rr))/(3*(tot2/(n*Rr))**2)
    a2=(tot2-S2)/((Rr-1)*n); a4=(tot4-S4)/((Rr-1)*n)
    Us=1-a4/(3*a2*a2)
    err=np.sqrt((Rr-1)/Rr*np.sum((Us-Us.mean())**2))
    return Uf,err
for T in [2.2,2.3,2.4]:
    t0=time.time()
    B2,B4,BA=run(T,seed=int(T*1000)+7,nequil=2000,nmeas=20000)
    np.savez(f'/work/T{T}.npz',B2=B2,B4=B4,BA=BA)
    Uf,err=jack(B2,B4)
    q=B2.shape[0]//4
    print(T,'U=%.6f err=%.6f'%(Uf,err),'<|m|> early %.5f late %.5f'%(BA[:q].mean(),BA[-q:].mean()),'time %.1f'%(time.time()-t0),flush=True)
