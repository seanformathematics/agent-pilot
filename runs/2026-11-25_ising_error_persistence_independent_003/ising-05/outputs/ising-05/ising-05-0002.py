
import numpy as np, time
N=16; NS=N*N
def make_nbrs():
    idx=np.arange(NS).reshape(N,N)
    right=np.roll(idx,-1,axis=1).ravel()  # neighbor to the right
    down=np.roll(idx,-1,axis=0).ravel()
    return right, down
RIGHT,DOWN=make_nbrs()
def sw_sweep(s, T, rng):
    # s: (K,NS) int8 spins +-1
    K=s.shape[0]
    p=1-np.exp(-2.0/T)
    same_r = s==s[:,RIGHT]
    same_d = s==s[:,DOWN]
    br = same_r & (rng.random((K,NS))<p)
    bd = same_d & (rng.random((K,NS))<p)
    # global flat labels
    off=(np.arange(K)*NS)[:,None]
    L=(np.arange(K*NS)).reshape(K,NS)
    # edges as flat indices
    src_r=np.nonzero(br.ravel())[0]
    dst_r=(src_r//NS)*NS+RIGHT[src_r%NS]
    src_d=np.nonzero(bd.ravel())[0]
    dst_d=(src_d//NS)*NS+DOWN[src_d%NS]
    a=np.concatenate([src_r,src_d]); b=np.concatenate([dst_r,dst_d])
    lab=np.arange(K*NS)
    while True:
        la=lab[a]; lb=lab[b]
        m=np.minimum(la,lb)
        new=lab.copy()
        np.minimum.at(new,a,m); np.minimum.at(new,b,m)
        # pointer jumping
        new=new[new]
        new=new[new]
        if np.array_equal(new,lab): break
        lab=new
    flip_lab = rng.random(K*NS)<0.5
    f = flip_lab[lab]
    return np.where(f, -s, s).astype(np.int8)
rng=np.random.default_rng(1)
K=400
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(K,NS))
t0=time.time()
T=2.2
ms=[]
for sweep in range(300):
    s=sw_sweep(s,T,rng)
    if sweep%20==0 or sweep<5:
        m=s.mean(axis=1)
        ms.append((sweep,np.mean(np.abs(m)),np.mean(m**2)))
print(time.time()-t0)
for x in ms: print(x)
