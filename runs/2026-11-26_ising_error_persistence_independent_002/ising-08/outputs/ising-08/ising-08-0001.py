
import numpy as np, time
N=16
def run(R, T, nsweep, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    acc=np.exp(-4.0/T*np.arange(0,1)) # placeholder
    p={d:np.exp(-2.0*d/T) for d in (4,8)}  # dE = 2*s*nn ; nn sum in {-4..4}
    def half(s,mask):
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nb  # energy change for flipping: 2*s*sum nn
        r=rng.random(s.shape)
        flip=(r<np.where(dE<=0,1.0,np.exp(-dE/T)))&mask
        s[flip]*=-1
    ms=[]
    for k in range(nsweep):
        half(s,masks[0]); half(s,masks[1])
        ms.append(s.reshape(R,-1).mean(1))
    return s,np.array(ms)
t=time.time()
s,ms=run(1000,2.269,50,1)
print(time.time()-t)
