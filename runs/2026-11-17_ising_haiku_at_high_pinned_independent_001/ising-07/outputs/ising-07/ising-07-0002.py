import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def make_masks(N):
    i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    return [((i+j)%2==p) for p in (0,1)]
MASKS=make_masks(N)
def run(T,R,nequil,nmeas,seed):
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    # acceptance table indexed by dE in {-8..8} for flips
    acc=np.zeros(17)
    for d in range(-8,9):
        acc[d+8]=1.0 if d<=0 else np.exp(-d/T)
    def sweep(S):
        for p in range(2):
            nn=np.roll(S,1,axis=1)+np.roll(S,-1,axis=1)+np.roll(S,1,axis=2)+np.roll(S,-1,axis=2)
            dE=2*S*nn  # energy change for flipping
            pr=acc[(dE+8)]
            r=rng.random(S.shape)
            flip=(r<pr)&MASKS[p]
            S[flip]*=-1
        return S
    for _ in range(nequil): S=sweep(S)
    m2=np.zeros(R); m4=np.zeros(R); cnt=0
    E=np.zeros(R)
    for _ in range(nmeas):
        S=sweep(S)
        m=S.reshape(R,-1).mean(axis=1)
        m2+=m*m; m4+=m**4; cnt+=1
    return m2/cnt, m4/cnt
def binder_jk(m2r,m4r):
    R=len(m2r)
    M2=m2r.mean(); M4=m4r.mean()
    U=1-M4/(3*M2**2)
    # jackknife over replicas
    s2=m2r.sum(); s4=m4r.sum()
    Uj=np.empty(R)
    for k in range(R):
        a2=(s2-m2r[k])/(R-1); a4=(s4-m4r[k])/(R-1)
        Uj[k]=1-a4/(3*a2**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
t=time.time()
m2r,m4r=run(2.269,1000,200,50,1)
print('time',time.time()-t)
print(binder_jk(m2r,m4r))
