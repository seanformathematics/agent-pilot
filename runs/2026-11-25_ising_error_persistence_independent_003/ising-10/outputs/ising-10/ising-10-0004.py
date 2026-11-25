import numpy as np, time, os
N=16; R=2000; NEQ=2000; NMEAS=8000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T,rng):
    for m in masks:
        nb = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE = 2*s*nb
        p = np.exp(-np.maximum(dE,0)/T)
        acc = ((dE<=0) | (rng.random(s.shape) < p)) & m[None]
        s[acc] *= -1
    return s
def run(T,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    for _ in range(NEQ): sweep(s,T,rng)
    a=np.zeros(R); b=np.zeros(R); absm=np.zeros(R)
    diag=[]
    for k in range(NMEAS):
        sweep(s,T,rng)
        m=s.reshape(R,-1).mean(axis=1)
        m2=m*m
        a+=m2; b+=m2*m2; absm+=np.abs(m)
        if k%1000==999: diag.append(float(np.abs(m).mean()))
    a/=NMEAS; b/=NMEAS; absm/=NMEAS
    np.savez(f'/work/T{T}.npz',a=a,b=b,absm=absm,diag=np.array(diag))
    U=1-b.mean()/(3*a.mean()**2)
    print(T,"U=",U,"<|m|>=",absm.mean(),"diag",np.round(diag,4),flush=True)
for T,seed in [(2.1,103),(2.2,104),(2.3,105),(2.4,106)]:
    t=time.time(); run(T,seed); print("time",time.time()-t,flush=True)