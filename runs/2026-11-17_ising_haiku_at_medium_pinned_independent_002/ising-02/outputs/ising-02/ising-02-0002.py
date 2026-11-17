
import numpy as np, time, sys
N=16
def make_masks():
    i,j=np.indices((N,N))
    return ((i+j)%2==0), ((i+j)%2==1)
MA,MB=make_masks()
def sweep(s,T,rng,tab):
    for M in (MA,MB):
        nb = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        # dE = 2*s*nb ; index = s*nb+4 in {0..8}
        idx = (s*nb+4)
        acc = rng.random(s.shape,dtype=np.float32) < tab[idx]
        flip = acc & M
        s[flip] *= -1
    return s
def run(T,R,neq,nmeas,seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    tab=np.zeros(9,dtype=np.float32)
    for k in range(9):
        dE=2*(k-4)  # dE = 2*s*nb where s*nb=k-4
        tab[k]=1.0 if dE<=0 else np.exp(-dE/T)
    for _ in range(neq): sweep(s,T,rng,tab)
    m2=np.zeros(R); m4=np.zeros(R)
    for _ in range(nmeas):
        sweep(s,T,rng,tab)
        m=s.reshape(R,-1).mean(1,dtype=np.float64)
        m2+=m*m; m4+=m**4
    return m2/nmeas, m4/nmeas
t=time.time()
a,b=run(2.269,1000,20,20,1)
print("time 40 sweeps R=1000:",time.time()-t)
