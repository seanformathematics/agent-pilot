import numpy as np, time, json, os
os.makedirs('/work',exist_ok=True)
N=16
def run(T, R, nburn, nmeas, seed, start):
    rng=np.random.default_rng(seed)
    if start=='cold':
        S=np.ones((R,N,N),dtype=np.int8)
    else:
        S=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    acc=np.zeros(9)
    for x in range(-4,5):
        dE=2*x
        acc[x+4]=1.0 if dE<=0 else np.exp(-dE/T)
    def sweep():
        for p in (0,1):
            nn=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.int8)
            a=acc[(S*nn).astype(np.int8)+4]
            flip=masks[p][None]&(rng.random((R,N,N))<a)
            S[flip]*=-1
    for _ in range(nburn): sweep()
    nblk=8; per=nmeas//nblk
    m2s=np.zeros((nblk,R)); m4s=np.zeros((nblk,R))
    for b in range(nblk):
        for _ in range(per):
            sweep()
            m=S.reshape(R,-1).mean(1)
            m2s[b]+=m*m; m4s[b]+=m**4
        m2s[b]/=per; m4s[b]/=per
    return m2s, m4s  # shape (nblk,R)

def U_and_err(m2r, m4r):
    # m2r, m4r: per-replica time averages, shape (R,)
    R=len(m2r)
    def U(idx):
        a=m2r[idx].mean(); b=m4r[idx].mean()
        return 1-b/(3*a*a)
    U0=U(np.arange(R))
    jk=np.empty(R)
    mask=np.ones(R,bool)
    S2=m2r.sum(); S4=m4r.sum()
    for k in range(R):
        a=(S2-m2r[k])/(R-1); b=(S4-m4r[k])/(R-1)
        jk[k]=1-b/(3*a*a)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return U0, err

Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
res={}
t0=time.time()
for T in Ts:
    start='cold' if T<2.269 else 'hot'
    seed=1000+int(round(T*100))
    m2b,m4b=run(T,2000,2000,4000,seed,start)
    m2=m2b.mean(0); m4=m4b.mean(0)
    U0,err=U_and_err(m2,m4)
    # drift: U per time block (using replica-averages within block)
    Ublk=[1-m4b[b].mean()/(3*m2b[b].mean()**2) for b in range(len(m2b))]
    res[T]=dict(U=float(U0),err=float(err),seed=seed,start=start,Ublk=[float(x) for x in Ublk])
    print(T,start,"U=%.5f err=%.5f"%(U0,err),"blocks:",np.round(Ublk,4),"t=%.0f"%(time.time()-t0),flush=True)
    json.dump(res,open('/work/res.json','w'))
    if time.time()-t0>480: break