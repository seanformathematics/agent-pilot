
import numpy as np, time, os
os.makedirs('/work', exist_ok=True)
N=16
def run(T, R, nsweep, nburn, seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    acc={dE:np.exp(-dE/T) for dE in (4,8)}
    def half(s,mask):
        h = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE = 2*s*h  # energy change if flipped (E=-sum ss)
        p = np.where(dE<=0,1.0,np.where(dE==4,acc[4],np.where(dE==8,acc[8],0.0)))
        flip = (rng.random(s.shape)<p) & mask
        s[flip] *= -1
    m2s=[];m4s=[]
    for it in range(nburn+nsweep):
        half(s,masks[0]); half(s,masks[1])
        if it>=nburn:
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2s.append(m*m); m4s.append(m**4)
    m2=np.array(m2s); m4=np.array(m4s)  # (nsweep,R)
    return m2,m4
t0=time.time()
m2,m4=run(2.3,256,300,200,1)
print('time',time.time()-t0)
# per-replica U from time averages
M2=m2.mean(0); M4=m4.mean(0)
Ur=1-M4/(3*M2**2)
print('U pooled',1-m4.mean()/(3*m2.mean()**2),'replica scatter/sqrt(R)',Ur.std(ddof=1)/np.sqrt(len(Ur)))
