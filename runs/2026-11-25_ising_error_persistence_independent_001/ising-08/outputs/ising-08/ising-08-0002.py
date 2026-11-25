import numpy as np, time
N=16
def make_masks():
    i,j=np.indices((N,N))
    return ((i+j)%2==0), ((i+j)%2==1)
def sweep(s, T, rng, masks, acc):
    # s: (R,N,N) int8 ; acc: lookup table of accept prob indexed by dE/2+4 ... dE in {-8..8} step 4
    for mk in masks:
        nb = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE = 2*s*nb  # energy change for flip
        p = acc[(dE//4)+2]  # dE in {-8,-4,0,4,8} -> idx 0..4
        u = rng.random(s.shape)
        flip = (u < p) & mk
        s[flip] *= -1
    return s
def make_acc(T):
    # dE values -8,-4,0,4,8 -> index (dE//4)+2
    dEs=np.array([-8,-4,0,4,8])
    return np.minimum(1.0,np.exp(-dEs/T))
masks=make_masks()
for R in [500,2000]:
    rng=np.random.default_rng(1)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    acc=make_acc(2.27)
    t=time.time()
    for _ in range(200): sweep(s,2.27,rng,masks,acc)
    print(R, (time.time()-t)/200*1e3,"ms/sweep")