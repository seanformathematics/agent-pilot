
import numpy as np, time, os, json
N=16
i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
MA=((i+j)%2==0); MB=~MA
def nbsum(s):
    return (np.roll(s,1,axis=1)+np.roll(s,-1,axis=1)+np.roll(s,1,axis=2)+np.roll(s,-1,axis=2))
def run(T,R,nburn,nmeas,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    tab=np.zeros(9)
    for x in [-4,-2,0,2,4]:
        dE=2*x; tab[x+4]=1.0 if dE<=0 else np.exp(-dE/T)
    A2=np.zeros(R); A4=np.zeros(R)
    # per-replica sums in two halves for drift check
    H2=np.zeros((2,R)); H4=np.zeros((2,R))
    for t in range(nburn+nmeas):
        for M in (MA,MB):
            nb=nbsum(s)
            x=(s.astype(np.int16)*nb.astype(np.int16))
            p=tab[x+4]
            flip=(rng.random((R,N,N))<p)&M
            s=np.where(flip,-s,s).astype(np.int8)
        if t>=nburn:
            m=s.reshape(R,-1).mean(axis=1)
            m2=m*m; m4=m2*m2
            A2+=m2; A4+=m4
            h=0 if (t-nburn)<nmeas//2 else 1
            H2[h]+=m2; H4[h]+=m4
    return A2/nmeas, A4/nmeas, H2/(nmeas/2), H4/(nmeas/2)
def U_of(a2,a4):
    return 1-a4.mean()/(3*a2.mean()**2)
def jack(a2,a4):
    R=len(a2); us=np.empty(R)
    S2=a2.sum(); S4=a4.sum()
    for k in range(R):
        us[k]=1-((S4-a4[k])/(R-1))/(3*((S2-a2[k])/(R-1))**2)
    return U_of(a2,a4), np.sqrt((R-1)/R*np.sum((us-us.mean())**2))
res={}
for T,seed in [(1.9,101),(2.0,102),(2.1,103)]:
    t0=time.time()
    a2,a4,H2,H4=run(T,2000,3000,6000,seed)
    U,err=jack(a2,a4)
    Uh=[U_of(H2[h],H4[h]) for h in range(2)]
    res[T]=dict(U=U,err=err,halves=Uh,seed=seed)
    print(T,'U=%.5f err=%.5f'%(U,err),'halves',Uh,'<m2>',a2.mean(),'time',time.time()-t0,flush=True)
json.dump({str(k):v for k,v in res.items()},open('/work/res_a.json','w'))
