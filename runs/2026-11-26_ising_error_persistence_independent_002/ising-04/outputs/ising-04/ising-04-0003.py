
import numpy as np, time, sys, os
os.makedirs('/work',exist_ok=True)
def run(T, R, nsweeps, seed, fname, meas_every=10, block=1000):
    N=16
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.indices((N,N))
    masks=[((ii+jj)%2==p) for p in (0,1)]
    accT=np.array([min(1.0,np.exp(-2*v/T)) for v in range(-4,5)],dtype=np.float32)
    nb=nsweeps//block
    S2=np.zeros((nb,R)); S4=np.zeros((nb,R)); cnt=np.zeros(nb)
    t0=time.time()
    for sw in range(nsweeps):
        for p in (0,1):
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            sh=(s*h).astype(np.int16)+4
            pr=accT[sh]
            flip=(rng.random((R,N,N),dtype=np.float32)<pr)&masks[p]
            s=np.where(flip,-s,s).astype(np.int8)
        if (sw+1)%meas_every==0:
            b=sw//block
            m=s.reshape(R,-1).sum(1)/(N*N)
            m2=m*m
            S2[b]+=m2; S4[b]+=m2*m2; cnt[b]+=1
    np.savez(fname,S2=S2,S4=S4,cnt=cnt,T=T,R=R,nsweeps=nsweeps,seed=seed,block=block,meas_every=meas_every)
    print('done',T,time.time()-t0)
run(2.2, 2000, 30000, 2201, '/work/T2.2_s2201.npz')
