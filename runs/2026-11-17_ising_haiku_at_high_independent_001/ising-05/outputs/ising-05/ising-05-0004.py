import numpy as np, time, json
N=16
def run(T,R,nequil,nmeas,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    p=np.array([min(1.0,np.exp(-2*(k-4)/T)) for k in range(9)])
    def sweep():
        for mk in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            k=(s*nb+4).astype(np.int8)
            acc=rng.random((R,N,N))<p[k]
            s[acc&mk]*=-1
    for _ in range(nequil): sweep()
    S2=np.zeros(R); S4=np.zeros(R); n=0
    for it in range(nmeas):
        sweep()
        m=s.sum(axis=(1,2))/(N*N); m2=m*m
        S2+=m2; S4+=m2*m2; n+=1
    return S2/n,S4/n
def ujk(a,b):
    R=len(a); U=1-np.mean(b)/(3*np.mean(a)**2)
    J=np.array([1-np.mean(np.delete(b,r))/(3*np.mean(np.delete(a,r))**2) for r in range(R)])
    return U,np.sqrt((R-1)/R*np.sum((J-J.mean())**2))
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
t0=time.time()
for T in Ts:
    a,b=run(T,400,2000,3000,seed=int(round(T*1000))+99991)
    U,e=ujk(a,b)
    print(T, round(U,5), round(e,6), round(time.time()-t0,1), flush=True)