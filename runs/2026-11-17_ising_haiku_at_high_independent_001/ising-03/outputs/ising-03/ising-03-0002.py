import numpy as np, time
N=16
def run(T, R=256, therm=1000, nsw=3000, seed=1):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.indices((N,N)); black=((ii+jj)%2==0)
    def sweep():
        for mask in (black,~black):
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE=2*s*nb
            acc=(dE<=0)|(rng.random((R,N,N),dtype=np.float32)<np.exp(-np.clip(dE,0,None)/T).astype(np.float32))
            flip=acc&mask
            s[flip]*=-1
    for _ in range(therm): sweep()
    m2=np.zeros(R); m4=np.zeros(R); ms=[]
    for k in range(nsw):
        sweep()
        m=s.reshape(R,-1).mean(1,dtype=np.float64)
        m2+=m*m; m4+=m**4
        if k%500==0: ms.append(np.abs(m).mean())
    m2/=nsw; m4/=nsw
    Ur=1-m4/(3*m2**2)
    return Ur, ms
for T in [2.269,1.5]:
    t=time.time()
    Ur,ms=run(T,nsw=2000)
    print(T,'time',round(time.time()-t,1),'U mean',Ur.mean(),'SE',Ur.std(ddof=1)/np.sqrt(len(Ur)),'std_r',Ur.std(ddof=1))
    print(' |m| trace',np.round(ms,3))
