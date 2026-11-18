
import numpy as np, time
N=16
def run(T, R, neq, nmeas, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    acc={d:np.exp(-d/T) for d in (4,8)}
    def sweep():
        for mk in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb  # energy change for flipping
            u=rng.random(s.shape)
            p=np.where(dE<=0,1.0,np.where(dE==4,acc[4],np.where(dE==8,acc[8],0.0)))
            flip=(u<p)&mk
            s[flip]*=-1
    for _ in range(neq): sweep()
    m2=np.zeros(R); m4=np.zeros(R); n=0
    for _ in range(nmeas):
        sweep()
        m=s.reshape(R,-1).mean(axis=1)
        m2+=m**2; m4+=m**4; n+=1
    return m2/n, m4/n
t=time.time()
m2,m4=run(2.3,500,50,50,1)
print(time.time()-t)
