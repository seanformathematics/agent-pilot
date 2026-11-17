
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16; R=400; T=2.269
rng=np.random.default_rng(12345)
S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(S,T):
    for par in (0,1):
        nn=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.int8)
        dE=2*S*nn  # energy change if flipped
        acc=(dE<=0)|(rng.random(S.shape)<np.exp(-np.clip(dE,0,None)/T))
        S[acc & masks[par]] *= -1
    return S
t=time.time()
for _ in range(200): sweep(S,T)
print("time per sweep (R=400):",(time.time()-t)/200)
print(os.cpu_count())
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
