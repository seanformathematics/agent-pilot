
import numpy as np, time
N=16
i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mA=((i+j)%2==0); mB=~mA
def run_series(R,T,nburn,nmeas,seed):
    rng=np.random.default_rng(seed)
    s = rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    p=np.array([1.0 if 2*x<=0 else np.exp(-2*x/T) for x in range(-4,5)])
    def sweep(s):
        for mask in (mA,mB):
            nn = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            x = (s*nn).astype(np.int8)+4
            acc = rng.random((R,N,N)) < p[x]
            s = np.where(acc & mask,-s,s).astype(np.int8)
        return s
    for _ in range(nburn): s=sweep(s)
    out=np.zeros((nmeas,3,R))
    for k in range(nmeas):
        s=sweep(s)
        m=s.mean(axis=(1,2))
        m2=m*m
        out[k,0]=np.abs(m); out[k,1]=m2; out[k,2]=m2*m2
    return out
def binder(o):
    m2=o[:,1].mean(); m4=o[:,2].mean()
    return 1-m4/(3*m2*m2)
def tau_int(x,maxlag=400):
    x=x-x.mean(); n=len(x); c0=np.dot(x,x)/n
    t=1.0
    for l in range(1,maxlag):
        c=np.dot(x[:-l],x[l:])/(n-l)/c0
        if c<0.05: break
        t+=2*c
    return t
for T in [2.27,1.5]:
    t0=time.time()
    o=run_series(1000,T,500,2000,11)
    print("T",T,"time",time.time()-t0)
    # time stability: U in blocks of 500 sweeps
    for b in range(4):
        print("  block",b,"U=",binder(o[b*500:(b+1)*500]))
    print("  U all=",binder(o))
    # tau for m2 averaged over replicas
    series=o[:,1,:].mean(axis=1)  # mean over replicas
    print("  tau_int m2 (replica-avg series, rough):",tau_int(o[:,1,0]), tau_int(o[:,1,1]))
    print("  <|m|>=",o[:,0].mean(), "U per-rep std:",np.std([1-(o[:,2,r].mean())/(3*o[:,1,r].mean()**2) for r in range(200)]))
