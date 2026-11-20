import numpy as np, time
N=16
def run(T, R, neq, nmeas, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p)[None] for p in (0,1)]
    # acceptance for dE = 2*s*nb, s*nb in {-4..4}
    acc={k: np.exp(-2*k/T) for k in (4,8)}
    def sweep():
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            sn=s*nb  # in {-4,-2,0,2,4}
            # dE=2*sn ; accept if sn<=0 or rand<exp(-4 sn/T)
            p=np.where(sn<=0,1.0,np.where(sn==2,np.exp(-4/T),np.exp(-8/T)))
            r=rng.random(s.shape)
            flip=m & (r<p)
            s[flip]*=-1
    t0=time.time()
    for _ in range(neq): sweep()
    m2s=[];m4s=[]
    for _ in range(nmeas):
        sweep()
        m=s.reshape(R,-1).mean(1)
        m2s.append(m**2); m4s.append(m**4)
    M2=np.array(m2s); M4=np.array(m4s)
    return M2,M4,time.time()-t0
t=time.time()
M2,M4,dt=run(2.2,500,200,200,1)
print("time",dt)
m2=M2.mean(); m4=M4.mean()
print("U pooled",1-m4/(3*m2**2))
