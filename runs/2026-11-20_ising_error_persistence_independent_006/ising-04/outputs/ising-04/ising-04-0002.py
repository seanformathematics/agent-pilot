import numpy as np, time
N=16; R=1000
rng=np.random.default_rng(12345)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
black=((ii+jj)%2==0)
S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
def sweep(S,T,rng):
    p=np.exp(-8.0/T*np.arange(0,5)) # placeholder
    for mask in (black,~black):
        nb=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.int16)
        dE=2*S*nb  # in {-8,-4,0,4,8}
        acc=np.exp(-np.maximum(dE,0)/T)
        flip=(rng.random(S.shape)<acc)&mask
        S=np.where(flip,-S,S).astype(np.int8)
    return S
t=time.time()
for k in range(50):
    S=sweep(S,2.269,rng)
print("per sweep (R=1000):",(time.time()-t)/50)
