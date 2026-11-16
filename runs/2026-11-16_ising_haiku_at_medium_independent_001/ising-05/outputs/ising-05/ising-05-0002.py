import numpy as np, json, time, os
os.makedirs('/work',exist_ok=True)
R,N=1000,16
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
def run(T,seed,neq=1500,nmeas=6000):
    rng=np.random.default_rng(seed)
    beta=1.0/T
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    # precompute acceptance table
    acc_tab={d: np.exp(-beta*d) for d in (4,8)}
    def sweep(s):
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nb
            p=np.where(dE<=0,1.0,np.where(dE==4,acc_tab[4],acc_tab[8])).astype(np.float32)
            flip=(rng.random(s.shape,dtype=np.float32)<p)&m
            s[flip]*=-1
    for _ in range(neq): sweep(s)
    m2=np.zeros((nmeas,R)); m4=np.zeros((nmeas,R))
    for k in range(nmeas):
        sweep(s)
        mag=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2[k]=mag**2; m4[k]=mag**4
    # per-chain time averages
    a2=m2.mean(0); a4=m4.mean(0)
    def U_of(idx):
        M2=a2[idx].mean(); M4=a4[idx].mean()
        return 1-M4/(3*M2**2)
    U=U_of(np.arange(R))
    # jackknife over chains
    jk=np.array([U_of(np.delete(np.arange(R),i)) for i in range(R)]) if False else None
    # fast jackknife via sums
    S2=a2.sum(); S4=a4.sum()
    jk=1-((S4-a4)/(R-1))/(3*((S2-a2)/(R-1))**2)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return U,err,float(np.mean(m2)),float(np.mean(m4))
out={}
if os.path.exists('/work/res.json'): out=json.load(open('/work/res.json'))
t0=time.time()
for i,T in enumerate([1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]):
    if str(T) in out: continue
    if time.time()-t0>400: break
    U,err,m2,m4=run(T,seed=1000+i)
    out[str(T)]=dict(U=float(U),err=float(err),seed=1000+i)
    json.dump(out,open('/work/res.json','w'))
    print(T,U,err,time.time()-t0,flush=True)
