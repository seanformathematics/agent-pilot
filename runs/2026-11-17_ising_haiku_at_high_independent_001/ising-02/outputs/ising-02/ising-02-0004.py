
import numpy as np, time
N=16
def run(T, R, neq, nmeas, every, seed):
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    acc={4:np.exp(-4/T),8:np.exp(-8/T)}
    def sweep(S):
        for p in (0,1):
            nn=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
            dE=2*S*nn
            prob=np.where(dE<=0,1.0,np.where(dE==4,acc[4],np.where(dE==8,acc[8],0.0)))
            flip=(rng.random(S.shape)<prob)&masks[p]
            S[flip]*=-1
        return S
    for _ in range(neq): S=sweep(S)
    m2=[];m4=[]
    for k in range(nmeas):
        for _ in range(every): S=sweep(S)
        m=S.reshape(R,-1).mean(1,dtype=np.float64)
        m2.append(m**2); m4.append(m**4)
    return np.array(m2),np.array(m4)
def jk(m2,m4):
    a=m2.mean(0); b=m4.mean(0); R=len(a)
    U=1-b.mean()/(3*a.mean()**2)
    tot_a=a.sum(); tot_b=b.sum()
    A=(tot_a-a)/(R-1); B=(tot_b-b)/(R-1)
    Uj=1-B/(3*A*A)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
temps=[2.2,2.3,2.4,2.5,2.6,2.7,2.8]
for i,T in enumerate(temps):
    t=time.time()
    m2,m4=run(T,1000,1000,2000,5,2000+i)
    np.savez(f'/work/T{T}.npz',m2=m2,m4=m4)
    U,e=jk(m2,m4)
    U1,_=jk(m2[:1000],m4[:1000]); U2,_=jk(m2[1000:],m4[1000:])
    print(T, round(U,6), e, 'halves', round(U1,5), round(U2,5), round(time.time()-t,1), flush=True)
