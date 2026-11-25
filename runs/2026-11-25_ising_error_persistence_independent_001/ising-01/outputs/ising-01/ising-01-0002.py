
import time, numpy as np, json
N=16; R=1000; NEQ=3000; NMEAS=7000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def run(T,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep():
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nb
            acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.maximum(dE,0)/T))
            s[acc&m]*=-1
    for _ in range(NEQ): sweep()
    m2=np.zeros(R); m4=np.zeros(R); cnt=0
    mabs=[]
    for _ in range(NMEAS):
        sweep()
        m=s.sum((1,2))/(N*N)
        m2+=m*m; m4+=m**4; cnt+=1
    return m2/cnt, m4/cnt
def binder(m2,m4):
    return 1-m4.mean()/(3*m2.mean()**2)
def jack(m2,m4):
    Ur=binder(m2,m4)
    Rn=len(m2); J=np.empty(Rn)
    S2=m2.sum(); S4=m4.sum()
    for k in range(Rn):
        a=(S4-m4[k])/(Rn-1); b=(S2-m2[k])/(Rn-1)
        J[k]=1-a/(3*b*b)
    err=np.sqrt((Rn-1)/Rn*np.sum((J-J.mean())**2))
    return Ur,err
out={}
for T,seed in [(1.9,101),(2.0,102),(2.1,103)]:
    t=time.time()
    m2,m4=run(T,seed)
    U,e=jack(m2,m4)
    out[T]=dict(U=U,err=e,m2=float(m2.mean()))
    print(T,U,e,float(m2.mean()),time.time()-t,flush=True)
    np.save(f"/work/m2_{T}.npy",m2); np.save(f"/work/m4_{T}.npy",m4)
