import numpy as np, time, os
os.makedirs('/work', exist_ok=True)
N=16
def run(T, R, nburn, nmeas, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==c)[None] for c in (0,1)]
    table=np.array([min(1.0,np.exp(-2*(k-4)/T)) for k in range(9)],dtype=np.float32)
    def sweep(s):
        for c in (0,1):
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            k=(s*nn+4).astype(np.int8)
            p=table[k]
            r=rng.random(s.shape,dtype=np.float32)
            flip=(r<p)&masks[c]
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nburn): s=sweep(s)
    m2=np.zeros(R); m4=np.zeros(R)
    for _ in range(nmeas):
        s=sweep(s)
        m=s.sum(axis=(1,2),dtype=np.float64)/(N*N)
        m2+=m*m; m4+=m**4
    return m2/nmeas, m4/nmeas
def U_jack(a2,a4,nb=50):
    # chain-level jackknife over R chains
    def U(x2,x4): return 1-x4.mean()/(3*x2.mean()**2)
    full=U(a2,a4)
    R=len(a2); idx=np.array_split(np.arange(R),nb)
    js=[]
    for g in idx:
        mask=np.ones(R,bool); mask[g]=False
        js.append(U(a2[mask],a4[mask]))
    js=np.array(js)
    err=np.sqrt((nb-1)/nb*np.sum((js-js.mean())**2))
    return full,err
t=time.time()
a2,a4=run(2.3,500,500,1000,101)
print("time",time.time()-t)
print(U_jack(a2,a4))