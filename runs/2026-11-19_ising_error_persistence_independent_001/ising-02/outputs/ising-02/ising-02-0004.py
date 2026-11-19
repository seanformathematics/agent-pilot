
import numpy as np, time, os
N=16; R=500
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def run(T,seed,burn=3000,meas=20000):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep(s):
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.float32)
            dE=2*s*nb
            p=np.exp(-np.clip(dE,0,None)/T).astype(np.float32)
            acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<p)
            s[acc&m]*=-1
    for _ in range(burn): sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); absm=np.zeros(R)
    check=[]
    for t in range(meas):
        sweep(s)
        m=s.reshape(R,-1).mean(1)
        m2=m*m
        S2+=m2; S4+=m2*m2; absm+=np.abs(m)
        if t%2000==1999: check.append(float(np.abs(m).mean()))
    return S2/meas,S4/meas,absm/meas,check
for T,seed in [(2.3,105),(2.4,106)]:
    t0=time.time()
    m2,m4,am,chk=run(T,seed)
    np.savez(f'/work/raw_T{T}.npz',m2=m2,m4=m4,am=am)
    U=1-m4.mean()/(3*m2.mean()**2)
    print(T,'U=',U,'<|m|>=',am.mean(),'check',np.round(chk,4).tolist(),'time',time.time()-t0,flush=True)
