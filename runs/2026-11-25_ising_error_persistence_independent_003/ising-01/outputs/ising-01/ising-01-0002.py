import numpy as np, time, os, json
os.makedirs('/work',exist_ok=True)
N=16
def run(T, R, nsweep, nburn, seed, block=200):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==c)[None] for c in (0,1)]
    def sweep():
        for c in (0,1):
            sf=s.astype(np.int8)
            nn=(np.roll(sf,1,1)+np.roll(sf,-1,1)+np.roll(sf,1,2)+np.roll(sf,-1,2))
            x=sf*nn  # in {-4,-2,0,2,4}
            pacc=np.exp(-2.0*x/T)
            acc=(rng.random((R,N,N))<pacc)&masks[c]
            s[acc]*=-1
    for _ in range(nburn): sweep()
    nb=nsweep//block
    S2=np.zeros(nb); S4=np.zeros(nb); cnt=np.zeros(nb)
    for b in range(nb):
        m2=np.zeros(R); m4=np.zeros(R)
        for k in range(block):
            sweep()
            m=s.reshape(R,-1).sum(1)/(N*N)
            m2+=m*m; m4+=m**4
        S2[b]=m2.sum(); S4[b]=m4.sum(); cnt[b]=R*block
    return S2,S4,cnt
def binder_jk(S2,S4,cnt):
    nb=len(S2)
    def U(a2,a4,c):
        m2=a2/c; m4=a4/c
        return 1-m4/(3*m2*m2)
    full=U(S2.sum(),S4.sum(),cnt.sum())
    jk=np.array([U(S2.sum()-S2[b],S4.sum()-S4[b],cnt.sum()-cnt[b]) for b in range(nb)])
    err=np.sqrt((nb-1)/nb*np.sum((jk-jk.mean())**2))
    return full,err
if __name__=="__main__":
    t=time.time()
    S2,S4,cnt=run(2.27,500,200,50,1,block=50)
    print("time",time.time()-t, binder_jk(S2,S4,cnt))
