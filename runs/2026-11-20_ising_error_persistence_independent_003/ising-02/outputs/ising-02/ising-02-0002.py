import numpy as np, time
N=16
def make(R,rng):
    return rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def sweep(s,T,rng,acc_tab):
    for p in (0,1):
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        idx=((s*nb)+4)//2  # s*nb in {-4,-2,0,2,4} -> 0..4
        pa=acc_tab[idx]
        r=rng.random(s.shape,dtype=np.float32)
        flip=masks[p]&(r<pa)
        s[flip]*=-1
    return s
def acc_table(T):
    # dE = 2*s*nb ; s*nb=k -> dE=2k
    k=np.array([-4,-2,0,2,4])
    dE=2*k
    return np.where(dE<=0,1.0,np.exp(-np.maximum(dE,0)/T)).astype(np.float32)
rng=np.random.default_rng(1)
R=500
for T in [2.269,1.9]:
    s=make(R,rng).astype(np.int8)
    at=acc_table(T)
    t0=time.time()
    hist=[]
    for sw in range(400):
        sweep(s,T,rng,at)
        if sw%20==0:
            m=s.reshape(R,-1).mean(1)
            hist.append((sw,np.mean(np.abs(m)),np.mean(m**2)))
    print('T',T,'time',time.time()-t0)
    for h in hist: print(h)