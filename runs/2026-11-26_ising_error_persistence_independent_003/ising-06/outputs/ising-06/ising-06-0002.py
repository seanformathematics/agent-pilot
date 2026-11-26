import numpy as np, time
N=16
def make_masks():
    i,j=np.indices((N,N))
    return ((i+j)%2==0), ((i+j)%2==1)
MA,MB=make_masks()
def sweep(s,T,rng,mask_list,acc):
    R=s.shape[0]
    for mask in mask_list:
        nn = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE = 2*s*nn  # int8 values in {-8..8}
        u = rng.random(s.shape,dtype=np.float32)
        flip = (u < acc[dE+8]) & mask
        s[flip] *= -1
    return s
def run(T,R,nequil,nmeas,seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    acc=np.zeros(17,dtype=np.float32)
    for dE in range(-8,9):
        acc[dE+8]= 1.0 if dE<=0 else np.exp(-dE/T)
    for _ in range(nequil): sweep(s,T,rng,(MA,MB),acc)
    m2s=[];m4s=[];es=[]
    for _ in range(nmeas):
        sweep(s,T,rng,(MA,MB),acc)
        m=s.reshape(R,-1).mean(1)
        m2s.append((m*m).mean()); m4s.append((m**4).mean())
    return np.array(m2s),np.array(m4s)
t=time.time()
m2,m4=run(2.3,500,200,200,1)
print("time",time.time()-t)
t=time.time()
m2,m4=run(2.3,500,0,1000,2)
print("time per 1000 sweeps R=500",time.time()-t)
U=1-m4/(3*m2**2)
print(U.mean())
def tau(x):
    x=x-x.mean(); n=len(x)
    f=np.fft.rfft(x,2*n); ac=np.fft.irfft(f*np.conj(f))[:n]; ac/=ac[0]
    s=0.5
    for k in range(1,n):
        s+=ac[k]
        if k>5*s: return s
    return s
print("tau m2 (sweeps)",tau(m2))