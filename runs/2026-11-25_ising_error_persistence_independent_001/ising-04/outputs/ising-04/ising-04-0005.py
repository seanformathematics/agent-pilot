import numpy as np, time, json, os
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
        if t%500==0 or t==neq-1:
            nb=np.roll(s,1,1)+np.roll(s,1,2)
            E_hist.append((t,float(np.mean(-(s*nb).sum(axis=(1,2))/(N*N)))))
    m2=[];m4=[]
    for t in range(nmeas):
        sweep(s)
        m=s.sum(axis=(1,2))/(N*N)
        m2.append(m**2); m4.append(m**4)
    return E_hist, np.array(m2), np.array(m4)
res={}
for T,seed in [(2.2,2204),(2.3,2305),(2.4,2406)]:
    t0=time.time()
    E,m2,m4=run(T,1000,2000,6000,seed)
    M2=m2.mean(0); M4=m4.mean(0); R=len(M2)
    U=1-M4.mean()/(3*M2.mean()**2)
    S2=M2.sum(); S4=M4.sum()
    jk=1-((S4-M4)/(R-1))/(3*((S2-M2)/(R-1))**2)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    res[T]=dict(U=float(U),err=float(err),E=E,t=time.time()-t0)
    print(T,"U",U,"err",err,"E",[(a,round(b,4)) for a,b in E],"time",round(time.time()-t0,1),flush=True)
json.dump({str(k):v for k,v in res.items()},open('/work/part2.json','w'))