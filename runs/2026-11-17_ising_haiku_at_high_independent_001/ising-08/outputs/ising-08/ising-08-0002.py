
import numpy as np, time
N=16
def run(T, R, nburn, nmeas, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    maskA=((ii+jj)%2==0)[None]; maskB=~maskA
    acc={dE:np.exp(-dE/T) for dE in (4,8)}
    def sweep(s):
        for mask in (maskA,maskB):
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE2=(s*nn)  # dE = 2*s*nn ; values -4..4 ; dE>0 when s*nn>0
            r=rng.random(s.shape)
            flip=mask & ((dE2<=0) | ((dE2==1)&(r<acc[4])) | ((dE2==2)&(r<acc[8])))
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nburn): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R)
    for _ in range(nmeas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(1)
        m2=m*m; S2+=m2; S4+=m2*m2
    m2=S2/nmeas; m4=S4/nmeas
    return m2,m4
def U_est(m2,m4):
    return 1-np.mean(m4)/(3*np.mean(m2)**2)
def jack(m2,m4):
    R=len(m2); U=U_est(m2,m4)
    idx=np.arange(R)
    Us=np.array([U_est(np.delete(m2,i),np.delete(m4,i)) for i in range(R)]) if R<=400 else None
    if Us is None:
        # block jackknife with 100 blocks
        B=100; blocks=np.array_split(np.arange(R),B)
        Us=np.array([U_est(np.delete(m2,b),np.delete(m4,b)) for b in blocks]); n=B
    else: n=R
    var=(n-1)/n*np.sum((Us-Us.mean())**2)
    return U, np.sqrt(var)
for T in [2.8,2.269]:
    t=time.time()
    m2,m4=run(T,400,300,300,1)
    U,e=jack(m2,m4)
    print(T,U,e,time.time()-t)
