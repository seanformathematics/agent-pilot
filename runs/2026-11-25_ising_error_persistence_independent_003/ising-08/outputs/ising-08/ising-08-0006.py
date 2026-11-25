import numpy as np, time, json
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
    for _ in range(nmeas):
        s=sweep(s)
        M=s.sum(axis=(1,2))/(N*N)
        M2=M*M
        S2+=M2; S4+=M2*M2
    m2=S2/nmeas; m4=S4/nmeas
    U=1-m4.mean()/(3*m2.mean()**2)
    tot2=m2.sum(); tot4=m4.sum()
    a2=(tot2-m2)/(R-1); a4=(tot4-m4)/(R-1)
    jk=1-a4/(3*a2*a2)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return U, err, m2.mean(), m4.mean()
res={}
for T,seed in [(2.3,1003),(2.4,1004),(2.2,2003)]:
    t=time.time()
    U,err,a2,a4=run(T, seed=seed)
    res[f"{T}_{seed}"]=dict(T=T,U=float(U),err=float(err),m2=float(a2),m4=float(a4),seed=seed)
    print(res[f"{T}_{seed}"], time.time()-t, flush=True)
json.dump(res,open('/work/res_b.json','w'))