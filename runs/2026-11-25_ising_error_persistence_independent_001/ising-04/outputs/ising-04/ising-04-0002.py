import numpy as np, time
N=16; R=1000; T=2.2
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
acc={4:np.exp(-4/T),8:np.exp(-8/T)}
def sweep(s):
    for k in range(2):
        nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=2*s*nb  # int8 ok, range -8..8
        p=np.where(dE<=0,1.0,np.where(dE==4,acc[4],np.where(dE==8,acc[8],0.0)))
        flip=(rng.random(s.shape)<p)&masks[k]
        s[flip]*=-1
    return s
t=time.time()
for _ in range(100): sweep(s)
print("time per sweep",(time.time()-t)/100)