
import numpy as np, time, os
N=16; R=512
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mA=((ii+jj)%2==0)[None]; mB=~mA
def run(T, seed, burn=3000, nmeas=4000, every=10):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    p=np.array([min(1.0,np.exp(-d/T)) for d in range(-8,9)])
    def sweep(s):
        for mask in (mA,mB):
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE=(2*s*nn).astype(np.int8)
            acc=rng.random(s.shape)< p[dE+8]
            s=np.where(mask&acc,-s,s).astype(np.int8)
        return s
    for _ in range(burn): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); S1=np.zeros(R)
    traj=[]
    for k in range(nmeas):
        for _ in range(every): s=sweep(s)
        m=s.reshape(R,-1).mean(1)
        S2+=m**2; S4+=m**4; S1+=np.abs(m)
        if k%400==0: traj.append((m**2).mean())
    return S2/nmeas,S4/nmeas,S1/nmeas,traj
for i,T in enumerate([2.0,2.1,2.2]):
    t=time.time()
    m2,m4,m1,traj=run(T,seed=1001+i)
    np.savez(f'/work/T{T}.npz',m2=m2,m4=m4,m1=m1)
    print("T",T,"time",round(time.time()-t,1),"traj",np.round(traj,4),"U",1-m4.mean()/(3*m2.mean()**2),flush=True)
