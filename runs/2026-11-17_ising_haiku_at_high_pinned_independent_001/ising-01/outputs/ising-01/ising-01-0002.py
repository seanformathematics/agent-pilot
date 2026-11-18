import numpy as np, time
rng = np.random.default_rng(1)
N=16
def setup(R,T,rng):
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    return s, masks
def sweep(s, masks, T, rng, probtab):
    for m in masks:
        nn = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        # dE = 2*s*nn ; index by s*nn in {-4,-2,0,2,4} -> (s*nn+4)//2
        sn = (s*nn).astype(np.int8)
        acc = rng.random(s.shape, dtype=np.float32) < probtab[(sn+4)//2]
        flip = acc & m
        s[flip] *= -1
    return s
R=1000
for T in [2.3]:
    s,masks = setup(R,T,rng)
    probtab = np.array([np.exp(-2*2*(k*2-4)/T) if False else 0 for k in range(5)],dtype=np.float32)
    # build table: index k=(sn+4)//2 where sn=s*nn in {-4,-2,0,2,4}; dE=2*sn
    sns = np.array([-4,-2,0,2,4])
    probtab = np.minimum(1.0, np.exp(-2*sns/T)).astype(np.float32)
    t0=time.time()
    for _ in range(50): sweep(s,masks,T,rng,probtab)
    print("time per sweep R=1000:", (time.time()-t0)/50)
