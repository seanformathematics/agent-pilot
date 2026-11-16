
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def run(T, R, nequil, nmeas, every, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    acc={d:np.exp(-2*d/T) for d in (2,4,8)}  # dE=2*s*h, positive values
    def sweep():
        for p in (0,1):
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int16)
            dE=2*s.astype(np.int16)*nn
            prob=np.where(dE<=0,1.0,np.exp(-np.maximum(dE,0)/T))
            u=rng.random(s.shape)
            flip=(u<prob)&masks[p]
            s[flip]*=-1
    for _ in range(nequil): sweep()
    M=[]
    for k in range(nmeas):
        for _ in range(every): sweep()
        M.append(s.reshape(R,-1).mean(axis=1))
    return np.array(M)  # (nmeas,R)
def U_est(M):
    m2=(M**2).mean(); m4=(M**4).mean()
    return 1-m4/(3*m2*m2)
for T in [1.5,2.269,2.8]:
    t0=time.time()
    M=run(T,256,300,200,5,1)
    dt=time.time()-t0
    Ur=[]
    rng=np.random.default_rng(0)
    # replica bootstrap
    R=M.shape[1]
    for b in range(200):
        idx=rng.integers(0,R,R)
        Mb=M[:,idx]
        Ur.append(U_est(Mb))
    print(T, U_est(M), np.std(Ur), dt)
