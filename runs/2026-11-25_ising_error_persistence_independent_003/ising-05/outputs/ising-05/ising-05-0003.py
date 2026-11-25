
import numpy as np, time
N=16; NS=N*N
idx=np.arange(NS).reshape(N,N)
RIGHT=np.roll(idx,-1,axis=1).ravel()
DOWN=np.roll(idx,-1,axis=0).ravel()
def sw_sweep(s, T, rng):
    K=s.shape[0]
    p=1-np.exp(-2.0/T)
    br = (s==s[:,RIGHT]) & (rng.random((K,NS))<p)
    bd = (s==s[:,DOWN]) & (rng.random((K,NS))<p)
    lab=np.arange(K*NS)
    src_r=np.nonzero(br.ravel())[0]
    dst_r=(src_r//NS)*NS+RIGHT[src_r%NS]
    src_d=np.nonzero(bd.ravel())[0]
    dst_d=(src_d//NS)*NS+DOWN[src_d%NS]
    a=np.concatenate([src_r,src_d]); b=np.concatenate([dst_r,dst_d])
    while True:
        m=np.minimum(lab[a],lab[b])
        new=lab.copy()
        np.minimum.at(new,a,m); np.minimum.at(new,b,m)
        new=new[new]; new=new[new]
        if np.array_equal(new,lab): break
        lab=new
    flip_lab = rng.random(K*NS)<0.5
    f = flip_lab[lab].reshape(K,NS)
    return np.where(f, -s, s).astype(np.int8)
rng=np.random.default_rng(1)
K=400
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(K,NS))
t0=time.time()
T=2.2
for sweep in range(301):
    s=sw_sweep(s,T,rng)
    if sweep%25==0:
        m=s.mean(axis=1)
        print(sweep, round(np.mean(np.abs(m)),4), round(np.mean(m**2),4), round(time.time()-t0,1))
print(time.time()-t0)
