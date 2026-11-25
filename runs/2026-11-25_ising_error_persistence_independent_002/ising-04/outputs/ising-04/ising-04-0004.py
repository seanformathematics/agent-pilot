import numpy as np, time, json
N=16; R=1000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def U_of(a2,a4): return 1-a4.mean()/(3*a2.mean()**2)
results=[]
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    t0=time.time()
    SEED=2024+int(round(T*100))
    rng=np.random.default_rng(SEED)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep(S):
        for m in masks:
            nb=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.int8)
            dE=2*S*nb
            pacc=np.exp(-np.maximum(dE,0)/T).astype(np.float32)
            acc=(dE<=0)|(rng.random(S.shape,dtype=np.float32)<pacc)
            S[acc&m]*=-1
    for _ in range(1000): sweep(S)
    MEAS=6000
    sm2=np.zeros(R); sm4=np.zeros(R)
    for k in range(MEAS):
        sweep(S)
        m=S.reshape(R,-1).mean(axis=1); m2=m*m
        sm2+=m2; sm4+=m2*m2
    m2r=sm2/MEAS; m4r=sm4/MEAS
    U=U_of(m2r,m4r)
    jk=np.array([U_of(np.delete(m2r,r),np.delete(m4r,r)) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    # also block-free check: split replicas into two halves
    Ua=U_of(m2r[:500],m4r[:500]); Ub=U_of(m2r[500:],m4r[500:])
    d=dict(T=T,U=float(U),err=float(err),Ua=float(Ua),Ub=float(Ub),m2=float(m2r.mean()),seed=SEED,time=time.time()-t0)
    print(json.dumps(d),flush=True)
    results.append(d)
json.dump(results,open("/work/results.json","w"),indent=1)