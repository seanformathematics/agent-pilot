
import numpy as np, time, itertools
def make_sweep(N,T,rng):
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    def sweep(s):
        for p in (0,1):
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nn
            prob=np.exp(-np.maximum(dE,0)/T)
            r=rng.random(s.shape,dtype=np.float32)
            s=np.where((r<prob)&masks[p],-s,s)
        return s
    return sweep
# validation N=4 exact
N=4;T=2.0
configs=[]
for bits in itertools.product([-1,1],repeat=16):
    configs.append(np.array(bits,dtype=float).reshape(4,4))
C=np.array(configs)
E=-(np.sum(C*np.roll(C,1,1),axis=(1,2))+np.sum(C*np.roll(C,1,2),axis=(1,2)))
w=np.exp(-E/T); w/=w.sum()
m=C.mean(axis=(1,2))
ex2=np.sum(w*m**2); ex4=np.sum(w*m**4)
print("exact U4:",1-ex4/(3*ex2**2))
rng=np.random.default_rng(7)
sw=make_sweep(4,T,rng)
R=2000
s=np.where(rng.random((R,4,4))<0.5,-1.0,1.0).astype(np.float32)
for _ in range(200): s=sw(s)
ms=[]
for _ in range(300):
    s=sw(s); ms.append(s.mean(axis=(1,2)))
ms=np.array(ms).ravel()
print("MC U4:",1-np.mean(ms**4)/(3*np.mean(ms**2)**2))
# pilot N=16
N=16
res={}
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    t=time.time()
    rng=np.random.default_rng(100+int(T*10))
    sw=make_sweep(N,T,rng)
    R=500
    s=np.ones((R,N,N),dtype=np.float32)
    nb=1000; nm=4000
    for _ in range(nb): s=sw(s)
    M=np.zeros((nm,R),dtype=np.float32)
    for k in range(nm):
        s=sw(s); M[k]=s.mean(axis=(1,2))
    res[T]=M
    m2=np.mean(M**2); m4=np.mean(M**4)
    # first vs second half
    def U(X): return 1-np.mean(X**4)/(3*np.mean(X**2)**2)
    print(T, "U=",round(float(U(M)),4), "U first half",round(float(U(M[:nm//2])),4),"second",round(float(U(M[nm//2:])),4),"time",round(time.time()-t,1))
    np.save(f"/work/pilot_{T}.npy",M) if __import__('os').path.isdir('/work') else None
