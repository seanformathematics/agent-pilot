import numpy as np, time
N=16
def make_masks():
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    return [((ii+jj)%2==0), ((ii+jj)%2==1)]
masks=make_masks()
def sweep(s,T,rng,acc):
    # s: (R,N,N) int8
    for m in masks:
        nb = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE = 2*s*nb  # in {-8,...,8}
        r = rng.random(s.shape,dtype=np.float32)
        flip = (dE<=0) | (r < acc[np.clip(dE,0,8)])
        flip &= m
        s[flip] *= -1
    return s
R=400
T=2.269
acc=np.zeros(9,dtype=np.float32)
for dE in range(9): acc[dE]=np.exp(-dE/T)
rng=np.random.default_rng(12345)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
t0=time.time()
for k in range(200): sweep(s,T,rng,acc)
print("time per sweep",(time.time()-t0)/200)
ms=[];
t0=time.time()
for k in range(3000):
    sweep(s,T,rng,acc)
    m=s.reshape(R,-1).mean(1)
    ms.append(m)
ms=np.array(ms)
print("time",time.time()-t0)
m2=ms**2
m2t=m2.mean(1)
np.save('/work/pilot.npy',ms) if False else None
# autocorrelation of m2 averaged over replicas
x=m2t-m2t.mean()
ac=[np.dot(x[:-l],x[l:])/np.dot(x,x) for l in range(0,200,10)]
print(np.round(ac,3))
m4=(ms**4).mean(0); m2b=m2.mean(0)
U=1-m4.mean()/(3*m2b.mean()**2)
print("U",U)
