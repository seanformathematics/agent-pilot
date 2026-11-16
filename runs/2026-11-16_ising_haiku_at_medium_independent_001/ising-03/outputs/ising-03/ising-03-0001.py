
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def run(T, R, nsweep, ntherm, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    acc={4:np.exp(-4/T),8:np.exp(-8/T)}
    m2s=[];m4s=[]
    def sweep(s):
        for mk in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE=2*s*nb  # energy change for flipping: 2*s*sum nb
            r=rng.random(s.shape)
            flip=mk[None]&((dE<=0)|(r<np.where(dE==4,acc[4],acc[8])))
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(ntherm): s=sweep(s)
    for _ in range(nsweep):
        s=sweep(s)
        m=s.reshape(R,-1).mean(1,dtype=np.float64)
        m2s.append(m*m); m4s.append(m**4)
    return np.array(m2s), np.array(m4s)
t=time.time()
m2,m4=run(2.27,500,100,0,1)
print("time per sweep R=500:",(time.time()-t)/100)
