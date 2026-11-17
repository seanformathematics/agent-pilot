
import numpy as np, time
N=16
def make_masks():
    i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    return ((i+j)%2==0), ((i+j)%2==1)
mA,mB=make_masks()
def run(R,T,nsweep,seed,init=None):
    rng=np.random.default_rng(seed)
    s = rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N)) if init is None else init
    # acceptance table indexed by s*nn+4 with nn in {-4..4} sums of 4 neighbours (even values)
    p=np.zeros(9)
    for x in range(-4,5):
        dE=2*x
        p[x+4]=1.0 if dE<=0 else np.exp(-dE/T)
    for sw in range(nsweep):
        for mask in (mA,mB):
            nn = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            x = (s*nn).astype(np.int8)+4
            acc = rng.random((R,N,N)) < p[x]
            flip = acc & mask
            s = np.where(flip,-s,s).astype(np.int8)
    return s
t=time.time()
s=run(1000,2.27,10,1)
print("time 10 sweeps R=1000:",time.time()-t)
