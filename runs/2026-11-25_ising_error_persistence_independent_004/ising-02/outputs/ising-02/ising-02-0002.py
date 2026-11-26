
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def make_masks(N):
    i,j=np.indices((N,N))
    return ((i+j)%2==0), ((i+j)%2==1)
MA,MB=make_masks(N)

def sweep(s, beta, rng, MA, MB):
    # s: (R,N,N) int8 in {-1,1}
    R=s.shape[0]
    for mask in (MA,MB):
        nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        # dE = 2*s*nb ; accept if dE<=0 or rand<exp(-beta dE)
        dE = 2*s*nb
        acc = np.exp(-beta*np.maximum(dE,0))  # dE>0 probability
        r = rng.random(s.shape)
        flip = (r < acc) & mask
        s[flip] *= -1
    return s

R=1000
rng=np.random.default_rng(1)
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
t=time.time()
for k in range(200):
    sweep(s,1/2.2,rng,MA,MB)
print("time per sweep ms", (time.time()-t)/200*1000)
