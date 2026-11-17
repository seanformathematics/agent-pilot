
import numpy as np, time, os, glob, json
for f in glob.glob('/work/*.npz'): os.remove(f)
print(os.listdir('/work'))
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
    a=np.zeros(R); b=np.zeros(R)
    for k in range(nmeas):
        for _ in range(every): S=sweep(S)
        m=S.reshape(R,-1).mean(1,dtype=np.float64)
        a+=m**2; b+=m**4
    return a/nmeas,b/nmeas
def jk(a,b):
    R=len(a)
    U=1-b.mean()/(3*a.mean()**2)
    A=(a.sum()-a)/(R-1); B=(b.sum()-b)/(R-1)
    Uj=1-B/(3*A*A)
    return U,np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
temps=[2.2,2.3,2.4,2.5,2.6,2.7,2.8]
for i,T in enumerate(temps):
    t=time.time()
    a,b=run(T,1000,1000,2000,5,2000+i)
    U,e=jk(a,b)
    U1,_=jk(a[:500],b[:500]); U2,_=jk(a[500:],b[500:])
    print(T, round(U,6), e, 'halves', round(U1,5), round(U2,5), round(time.time()-t,1), flush=True)
