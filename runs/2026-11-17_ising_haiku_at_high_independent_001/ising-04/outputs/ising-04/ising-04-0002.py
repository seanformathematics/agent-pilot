
import numpy as np, time
N=16; R=400
def setup(seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    return rng,s,masks
def sweep(s,T,rng,masks):
    for mk in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nb  # energy change for flipping s (E=-sum s s)
        acc=np.exp(-dE/T)
        u=rng.random(s.shape,dtype=np.float32)
        flip=(u<acc)&mk
        s[flip]*=-1
    return s
def run(T,nsweep,ntherm,seed):
    rng,s,masks=setup(seed)
    for _ in range(ntherm): sweep(s,T,rng,masks)
    m2s=[];m4s=[]
    for _ in range(nsweep):
        sweep(s,T,rng,masks)
        m=s.reshape(R,-1).mean(1)
        m2s.append(m**2); m4s.append(m**4)
    m2=np.array(m2s); m4=np.array(m4s)
    return m2,m4
t0=time.time()
m2,m4=run(2.269,200,200,1)
print('time',time.time()-t0)
for T in [1.5,2.269,2.8]:
    m2,m4=run(T,300,300,2)
    M2=m2.mean(); M4=m4.mean()
    print(T, 1-M4/(3*M2**2))
print(time.time())
