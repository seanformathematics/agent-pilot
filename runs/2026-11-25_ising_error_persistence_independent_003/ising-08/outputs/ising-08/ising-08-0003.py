import numpy as np, time, json, sys
N=16
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mask0=((ii+jj)%2==0); mask1=~mask0
def run(T, seed, R=1000, neq=3000, nmeas=10000):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep(s):
        for mask in (mask0,mask1):
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nb
            acc=np.exp(-np.maximum(dE,0)/T)
            flip=(rng.random(s.shape)<acc)&mask
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(neq): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R)
    Mhist=[]
    for _ in range(nmeas):
        s=sweep(s)
        M=s.sum(axis=(1,2))/(N*N)
        M2=M*M
        S2+=M2; S4+=M2*M2
    m2=S2/nmeas; m4=S4/nmeas
    # ratio estimator and jackknife over replicas
    def U_of(a2,a4):
        return 1-a4.mean()/(3*a2.mean()**2)
    U=U_of(m2,m4)
    jk=np.empty(R)
    tot2=m2.sum(); tot4=m4.sum()
    for r in range(R):
        a2=(tot2-m2[r])/(R-1); a4=(tot4-m4[r])/(R-1)
        jk[r]=1-a4/(3*a2*a2)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    # split-half check
    return U, err, m2.mean(), m4.mean()
if __name__=="__main__":
    T=float(sys.argv[1]); idx=int(sys.argv[2])
    t=time.time()
    U,err,a2,a4=run(T, seed=1000+idx)
    out=dict(T=T,U=U,err=err,m2=a2,m4=a4,seed=1000+idx,time=time.time()-t)
    print(json.dumps(out))
    with open(f"/work/res_{T}.json","w") as f: json.dump(out,f)
