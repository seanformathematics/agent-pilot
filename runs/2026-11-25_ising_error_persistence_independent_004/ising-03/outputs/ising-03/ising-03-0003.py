import numpy as np, time
N=16; R=512
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
def run(T,neq,nmeas,seed,block=10):
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
    m2s=[];m4s=[]
    for _ in range(nmeas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(1)
        m2s.append(m*m); m4s.append(m**4)
    return np.array(m2s), np.array(m4s)
def U_est(m2,m4):
    # m2,m4: (nmeas,R) -> per replica time averages
    a2=m2.mean(0); a4=m4.mean(0)
    Ur=1-a4/(3*a2**2)
    return Ur.mean(), Ur.std(ddof=1)/np.sqrt(len(Ur))
out={}
t0=time.time()
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    m2,m4=run(T,1000,4000,seed=int(T*1000))
    U,e=U_est(m2,m4)
    # crude autocorrelation of m2 over replicas-averaged series
    x=m2.mean(1)-m2.mean()
    ac=[np.dot(x[:-k],x[k:])/np.dot(x,x) if k>0 else 1 for k in range(0,200)]
    tau=1+2*sum(ac[1:next((k for k in range(1,200) if ac[k]<0.05),199)])
    out[T]=(U,e)
    print(T, round(U,5), round(e,5), "tau_int(m2)~",round(tau,1), "time",round(time.time()-t0,1), flush=True)
