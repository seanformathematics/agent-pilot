import numpy as np, time
N=16
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def run(T,R,neq,nmeas,seed):
    rng=np.random.default_rng(seed)
    acc={4:np.exp(-4/T),8:np.exp(-8/T)}
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep(s):
        for k in range(2):
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb
            p=np.where(dE<=0,1.0,np.where(dE==4,acc[4],np.where(dE==8,acc[8],0.0)))
            flip=(rng.random(s.shape)<p)&masks[k]
            s[flip]*=-1
    E_hist=[]
    for t in range(neq):
        sweep(s)
        if t%200==0:
            nb=np.roll(s,1,1)+np.roll(s,1,2)
            E_hist.append((t,float(np.mean(-(s*nb).sum(axis=(1,2))/(N*N)))))
    m1=[];m2=[];m4=[]
    for t in range(nmeas):
        sweep(s)
        m=s.sum(axis=(1,2))/(N*N)
        m1.append(np.abs(m)); m2.append(m**2); m4.append(m**4)
    m2=np.array(m2); m4=np.array(m4)  # (nmeas,R)
    return E_hist, m2, m4
for T in [2.3,2.1]:
    t0=time.time()
    E,m2,m4=run(T,500,1000,4000,11)
    print("T",T,"E hist",[(a,round(b,3)) for a,b in E])
    # per-replica averages
    M2=m2.mean(0); M4=m4.mean(0)
    U=1-M4.mean()/(3*M2.mean()**2)
    # jackknife over replicas
    R=len(M2); jk=[]
    for r in range(R):
        mask=np.ones(R,bool); mask[r]=False
        jk.append(1-M4[mask].mean()/(3*M2[mask].mean()**2))
    jk=np.array(jk)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    # autocorrelation of m2 time series averaged over replicas
    x=m2.mean(1)-m2.mean()
    ac=[np.dot(x[:-l],x[l:])/np.dot(x,x) for l in [1,5,10,20,50]]
    print("U",U,"err",err,"ac",np.round(ac,3),"time",time.time()-t0)