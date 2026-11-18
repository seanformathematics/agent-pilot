import numpy as np, time, json
N=16
i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
MASKS=[((i+j)%2==p) for p in (0,1)]
def run(T,R,nequil,nmeas,seed):
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    acc=np.array([1.0 if d<=0 else np.exp(-d/T) for d in range(-8,9)])
    def sweep(S):
        for p in range(2):
            nn=np.roll(S,1,axis=1)+np.roll(S,-1,axis=1)+np.roll(S,1,axis=2)+np.roll(S,-1,axis=2)
            dE=2*S*nn
            pr=acc[(dE+8)]
            r=rng.random(S.shape)
            flip=(r<pr)&MASKS[p]
            S[flip]*=-1
        return S
    for _ in range(nequil): S=sweep(S)
    m2=np.zeros(R); m4=np.zeros(R); cnt=0
    trace=[]
    for k in range(nmeas):
        S=sweep(S)
        m=S.reshape(R,-1).mean(axis=1)
        m2+=m*m; m4+=m**4; cnt+=1
        if k%200==0: trace.append(float(np.mean(m*m)))
    return m2/cnt, m4/cnt, trace
def binder_jk(m2r,m4r):
    R=len(m2r)
    M2=m2r.mean(); M4=m4r.mean()
    U=1-M4/(3*M2**2)
    s2=m2r.sum(); s4=m4r.sum()
    a2=(s2-m2r)/(R-1); a4=(s4-m4r)/(R-1)
    Uj=1-a4/(3*a2**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
out={}
t0=time.time()
for idx,T in enumerate(Ts):
    m2r,m4r,tr=run(T,1000,2000,2000,1000+idx)
    U,e=binder_jk(m2r,m4r)
    out[T]=dict(U=float(U),err=float(e),trace=tr)
    print(T, round(U,5), round(e,5), 'trace m2 first/last', round(tr[0],4), round(tr[-1],4), round(time.time()-t0,1), flush=True)
json.dump(out,open('/work/prod1.json','w'))
