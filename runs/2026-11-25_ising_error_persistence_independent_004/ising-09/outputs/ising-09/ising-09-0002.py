import numpy as np, time
N=16
def make_mask():
    i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    return ((i+j)%2==0), ((i+j)%2==1)
A,B=make_mask()
def sweep(s,T,rng,maskA,maskB):
    # s shape (R,N,N) int8
    for mask in (maskA,maskB):
        h = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        # flip prob exp(-2 s h /T) for sites in mask
        dE = 2*s*h  # energy change *(T) ; positive means cost
        p = np.exp(-dE/T)
        r = rng.random(s.shape,dtype=np.float32)
        flip = mask & (r < p)
        s = np.where(flip, -s, s).astype(np.int8)
    return s
rng=np.random.default_rng(12345)
R=400
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
t=time.time()
for k in range(20):
    s=sweep(s,2.269,rng,A,B)
print("time per sweep (R=400):",(time.time()-t)/20)
