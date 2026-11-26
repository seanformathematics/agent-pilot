import numpy as np, time
N=16
Ts=np.array([1.9,2.0,2.1,2.2,2.3,2.4])
def make_masks(N):
    i,j=np.indices((N,N))
    return [((i+j)%2==p) for p in (0,1)]
masks=make_masks(N)
def sweep(s, beta, rng, masks):
    # s: (R,N,N) int8 ; beta: (R,1,1) float32
    for m in masks:
        nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE = 2*s*nb  # int8 in {-8..8}
        acc = np.exp(-beta*np.maximum(dE,0).astype(np.float32))
        r = rng.random(s.shape,dtype=np.float32)
        flip = (r<acc) & m
        s[flip] *= -1
    return s
rng=np.random.default_rng(12345)
R=6*1000
T=np.repeat(Ts,1000)
beta=(1.0/T).astype(np.float32).reshape(-1,1,1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
t0=time.time()
for k in range(10): sweep(s,beta,rng,masks)
print("time per sweep (6000 replicas):",(time.time()-t0)/10)
m=s.reshape(R,-1).mean(1)
print(m[:5])