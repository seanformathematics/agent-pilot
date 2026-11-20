import numpy as np, time, json
N=16
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def acc_table(T):
    k=np.array([-4,-2,0,2,4]); dE=2*k
    return np.where(dE<=0,1.0,np.exp(-np.maximum(dE,0)/T)).astype(np.float32)
def sweep(s,rng,at):
    for p in (0,1):
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        pa=at[((s*nb)+4)//2]
        r=rng.random(s.shape,dtype=np.float32)
        s[masks[p]&(r<pa)]*=-1
def run(T,R,burn,meas,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    at=acc_table(T)
    for _ in range(burn): sweep(s,rng,at)
    M2=np.zeros((meas,R)); M4=np.zeros((meas,R))
    for t in range(meas):
        sweep(s,rng,at)
        m=s.reshape(R,-1).mean(1,dtype=np.float64)
        M2[t]=m*m; M4[t]=m**4
    return M2,M4
def Ufun(m2,m4): return 1-np.mean(m4)/(3*np.mean(m2)**2)
def jk(m2,m4,G=20):
    R=len(m2); idx=np.arange(R)%G
    U=Ufun(m2,m4)
    Us=np.array([Ufun(m2[idx!=g],m4[idx!=g]) for g in range(G)])
    return U, np.sqrt((G-1)/G*np.sum((Us-Us.mean())**2))
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
seeds={1.9:101,2.0:102,2.1:103,2.2:104,2.3:105,2.4:106}
out={}
t0=time.time()
for T in Ts:
    M2,M4=run(T,400,2000,5000,seeds[T])
    m2=M2.mean(0); m4=M4.mean(0)
    U,e=jk(m2,m4)
    h=M2.shape[0]//2
    U1=Ufun(M2[:h].mean(0),M4[:h].mean(0)); U2=Ufun(M2[h:].mean(0),M4[h:].mean(0))
    out[T]=dict(U=U,err=e,n=400*5000)
    print(T,'U=%.5f err=%.5f'%(U,e),'halves %.5f %.5f'%(U1,U2),'t',round(time.time()-t0,1),flush=True)
json.dump({str(k):v for k,v in out.items()},open('/work/results.json','w'))