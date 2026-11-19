
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def make_masks(N):
    i,j=np.indices((N,N))
    return [( (i+j)%2==p) for p in (0,1)]
MASKS=make_masks(N)
def sweep(s,T,rng,masks):
    # s: (R,N,N) int8
    for mask in masks:
        S = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE = 2*s*S  # in {-8..8}
        p = np.exp(-np.maximum(dE,0)/T)  # acceptance for dE>0; dE<=0 always accept
        u = rng.random(s.shape)
        acc = (dE<=0) | (u<p)
        s[acc & mask] *= -1
    return s
rng=np.random.default_rng(12345)
R=1000
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
t0=time.time()
for k in range(200):
    sweep(s,2.27,rng,MASKS)
print("time per sweep (R=1000):",(time.time()-t0)/200)
