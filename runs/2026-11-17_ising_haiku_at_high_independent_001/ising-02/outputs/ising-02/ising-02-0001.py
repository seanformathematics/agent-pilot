
import numpy as np, time, os
os.makedirs('/work', exist_ok=True)
N=16
def run(T, R, neq, nmeas, every, seed):
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    acc={d:np.exp(-d/T) for d in (4,8)}
    def sweep(S):
        for p in (0,1):
            nn=(np.roll(S,1,1).astype(np.int8)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
            dE=2*S*nn  # in {-8..8}
            prob=np.where(dE<=0,1.0,np.where(dE==4,acc[4],np.where(dE==8,acc[8],0.0)))
            u=rng.random(S.shape)
            flip=(u<prob)&masks[p]
            S[flip]*=-1
        return S
    for _ in range(neq): S=sweep(S)
    m2=[];m4=[]
    for k in range(nmeas):
        for _ in range(every): S=sweep(S)
        m=S.reshape(R,-1).mean(1,dtype=np.float64)
        m2.append(m**2); m4.append(m**4)
    m2=np.array(m2); m4=np.array(m4)  # (nmeas,R)
    return m2,m4
t=time.time()
m2,m4=run(2.269,1000,200,50,2,1)
print(time.time()-t)
M2=m2.mean(); M4=m4.mean()
print(1-M4/(3*M2**2))
