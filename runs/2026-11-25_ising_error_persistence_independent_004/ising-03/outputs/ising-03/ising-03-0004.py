import numpy as np, time, json
N=16; R=512
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
def run(T,neq,nmeas,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep(s):
        for c in (0,1):
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            p=np.exp(-(2*s*nn)/T)
            acc=(rng.random(s.shape)<p)&masks[c]
            s=np.where(acc,-s,s).astype(np.int8)
        return s
    for _ in range(neq): s=sweep(s)
    a2=np.zeros(R); a4=np.zeros(R)
    for _ in range(nmeas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(1)
        a2+=m*m; a4+=m**4
    return a2/nmeas, a4/nmeas   # per-replica time averages
res={}
t0=time.time()
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    a2,a4=run(T,1500,6000,seed=777+int(T*1000))
    Ur=1-a4/(3*a2**2)
    U=Ur.mean(); e=Ur.std(ddof=1)/np.sqrt(R)
    res[T]=(float(U),float(e))
    print(T, round(U,5), round(e,5), round(time.time()-t0,1), flush=True)
json.dump(res,open('/work/batch2.json','w'))
