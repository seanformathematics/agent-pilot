
import numpy as np, time, json, os
N=16; R=2000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,beta,rng):
    for m in masks:
        nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=2*s*nb
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-beta*np.clip(dE,0,None)))
        s[acc&m]*=-1
    return s
def run(T,seed,neq=2000,nmeas=4000):
    rng=np.random.default_rng(seed)
    beta=1/T
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    for _ in range(neq): sweep(s,beta,rng)
    S2=np.zeros(R); S4=np.zeros(R)
    h2=None; h4=None
    for t in range(nmeas):
        sweep(s,beta,rng)
        m=s.reshape(R,-1).sum(1)/(N*N)
        m2=m*m
        S2+=m2; S4+=m2*m2
        if t==nmeas//2-1: h2=S2.copy(); h4=S4.copy()
    def U_of(a2,a4,nn):
        M2=a2/nn; M4=a4/nn
        return 1-M4/(3*M2*M2)
    nn=R*nmeas
    U=U_of(S2.sum(),S4.sum(),nn)
    tot2=S2.sum(); tot4=S4.sum()
    Ujk=np.array([U_of(tot2-S2[r],tot4-S4[r],nn-nmeas) for r in range(R)])
    var=(R-1)/R*np.sum((Ujk-Ujk.mean())**2)
    Uh1=U_of(h2.sum(),h4.sum(),R*nmeas//2)
    Uh2=U_of(tot2-h2.sum(),tot4-h4.sum(),R*nmeas//2)
    return U,np.sqrt(var),Uh1,Uh2
res={}
for T,seed in [(2.2,302),(2.3,303),(2.4,304)]:
    t0=time.time()
    U,Ue,h1,h2=run(T,seed)
    res[T]=dict(U=U,err=Ue,half1=h1,half2=h2,seed=seed)
    print(T,U,Ue,h1,h2,time.time()-t0,flush=True)
json.dump(res,open('/work/res_c.json','w'))
