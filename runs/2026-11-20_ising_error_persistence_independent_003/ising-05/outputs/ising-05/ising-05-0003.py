import numpy as np, time
N=16
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0), ((ii+jj)%2==1)]
def sweep(s,rng,acc):
    for m in masks:
        nb = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE = 2*s*nb
        r = rng.random(s.shape,dtype=np.float32)
        flip = ((dE<=0) | (r < acc[np.clip(dE,0,8)])) & m
        s[flip] *= -1
def run(T,R=400,neq=1000,nmeas=3000,seed=1):
    acc=np.array([np.exp(-dE/T) for dE in range(9)],dtype=np.float32)
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    for k in range(neq): sweep(s,rng,acc)
    ms=np.empty((nmeas,R))
    for k in range(nmeas):
        sweep(s,rng,acc)
        ms[k]=s.reshape(R,-1).mean(1)
    return ms
for T in [2.269,2.0]:
    ms=run(T)
    m2=(ms**2)
    x=m2.mean(1); x=x-x.mean()
    ac=[]
    for l in [0,1,2,5,10,20,40,80]:
        ac.append(np.dot(x[:len(x)-l],x[l:])/np.dot(x,x)*len(x)/(len(x)-l) if l>0 else 1.0)
    print(T,np.round(ac,3))
    m2b=m2.mean(0); m4b=(ms**4).mean(0)
    U=1-m4b.mean()/(3*m2b.mean()**2)
    print("U",U, "m2",m2b.mean())
