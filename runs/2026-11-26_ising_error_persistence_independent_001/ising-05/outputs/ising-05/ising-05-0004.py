import numpy as np, time
N=16; R=1000
def run(T, nsweep, neq, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    out=np.zeros((nsweep,3))
    for t in range(neq+nsweep):
        for m in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb
            p=np.exp(-dE/T)
            flip=(rng.random(s.shape)<p)&m[None]
            s[flip]*=-1
        if t>=neq:
            mag=s.reshape(R,-1).mean(1,dtype=np.float64)
            a=mag**2
            out[t-neq]=[a.mean(), (a**2).mean(), np.abs(mag).mean()]
    return out
t0=time.time()
for T,seed in [(2.1,103),(2.2,104)]:
    o=run(T,20000,2000,seed)
    np.save(f'/work/obs_T{T}_N16.npy',o)
    m2=o[:,0].mean(); m4=o[:,1].mean()
    print(T, time.time()-t0, 1-m4/(3*m2**2), flush=True)