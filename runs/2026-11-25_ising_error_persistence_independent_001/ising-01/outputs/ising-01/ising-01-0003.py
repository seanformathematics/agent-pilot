
import time, numpy as np
N=16; R=1000; NEQ=5000; NMEAS=7000
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
    H=[]
    for h in range(2):
        m2=np.zeros(R); m4=np.zeros(R)
        for _ in range(NMEAS//2):
            sweep()
            m=s.sum((1,2))/(N*N)
            m2+=m*m; m4+=m**4
        H.append((m2/(NMEAS//2), m4/(NMEAS//2)))
    return H
def binder(m2,m4): return 1-m4.mean()/(3*m2.mean()**2)
def jack(m2,m4):
    Ur=binder(m2,m4); Rn=len(m2); J=np.empty(Rn)
    S2=m2.sum(); S4=m4.sum()
    for k in range(Rn):
        a=(S4-m4[k])/(Rn-1); b=(S2-m2[k])/(Rn-1); J[k]=1-a/(3*b*b)
    return Ur, np.sqrt((Rn-1)/Rn*np.sum((J-J.mean())**2))
for T,seed in [(2.2,104),(2.3,105),(2.4,106)]:
    t=time.time()
    H=run(T,seed)
    m2=np.concatenate([H[0][0],H[1][0]]); m4=np.concatenate([H[0][1],H[1][1]])
    U,e=jack(m2,m4)
    U1=binder(H[0][0],H[0][1]); U2=binder(H[1][0],H[1][1])
    print(T,"U=",U,"err=",e,"m2=",m2.mean(),"halves:",U1,U2,round(time.time()-t,1),flush=True)
    np.save(f"/work/m2_{T}.npy",m2); np.save(f"/work/m4_{T}.npy",m4)
