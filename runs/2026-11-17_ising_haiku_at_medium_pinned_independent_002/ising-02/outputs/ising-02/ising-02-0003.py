
import numpy as np, time, json, os
N=16
i,j=np.indices((N,N))
MA=((i+j)%2==0); MB=~MA
def sweep(s,rng,tab):
    for M in (MA,MB):
        nb = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        acc = rng.random(s.shape,dtype=np.float32) < tab[s*nb+4]
        s[acc & M] *= -1
def run(T,R,neq,nmeas,seed,start):
    rng=np.random.default_rng(seed)
    if start=='cold': s=np.ones((R,N,N),dtype=np.int8)
    else: s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    tab=np.array([1.0 if 2*(k-4)<=0 else np.exp(-2*(k-4)/T) for k in range(9)],dtype=np.float32)
    for _ in range(neq): sweep(s,rng,tab)
    m2=np.zeros(R); m4=np.zeros(R); ms=[]
    for _ in range(nmeas):
        sweep(s,rng,tab)
        m=s.reshape(R,-1).mean(1,dtype=np.float64)
        m2+=m*m; m4+=m**4
    return m2/nmeas, m4/nmeas
def U_of(m2,m4):
    return 1-m4.mean()/(3*m2.mean()**2)
def jack(m2,m4):
    R=len(m2); u=U_of(m2,m4)
    loo=np.array([U_of(np.delete(m2,r),np.delete(m4,r)) for r in range(R)])
    return u, np.sqrt((R-1)/R*np.sum((loo-loo.mean())**2))
out={}
t0=time.time()
for T in [1.5,1.6,1.7,1.8,1.9,2.0,2.1]:
    start='cold' if T<2.269 else 'hot'
    m2,m4=run(T,1000,2000,6000,int(1000*T),start)
    u,e=jack(m2,m4)
    out[T]=(u,e)
    print(T,u,e,time.time()-t0,flush=True)
json.dump({str(k):v for k,v in out.items()},open('/work/resA.json','w'))
