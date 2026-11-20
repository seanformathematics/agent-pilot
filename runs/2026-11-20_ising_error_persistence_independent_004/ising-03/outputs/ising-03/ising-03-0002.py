
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16; R=1000; BURN=3000; MEAS=5000
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def run(T,seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    # acceptance table for dE=2*s*h, h in {-4,-2,0,2,4}
    def sweep(s):
        for m in masks:
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s.astype(np.int16)*h
            p=np.exp(-np.maximum(dE,0)/T)
            acc=(rng.random(s.shape)<p)&m
            s=np.where(acc,-s,s).astype(np.int8)
        return s
    t0=time.time()
    absm_hist=[]
    for k in range(BURN):
        s=sweep(s)
        if k%500==499: absm_hist.append(np.abs(s.mean(axis=(1,2))).mean())
    S2=np.zeros(R); S4=np.zeros(R); S1=np.zeros(R)
    for k in range(MEAS):
        s=sweep(s)
        m=s.mean(axis=(1,2))
        m2=m*m
        S2+=m2; S4+=m2*m2; S1+=np.abs(m)
    return dict(S2=S2,S4=S4,S1=S1,hist=np.array(absm_hist),time=time.time()-t0)
for idx,T in enumerate(Ts):
    r=run(T,seed=1000+idx)
    np.savez(f'/work/run_T{T}.npz',S2=r['S2'],S4=r['S4'],S1=r['S1'],hist=r['hist'],T=T,seed=1000+idx,meas=MEAS)
    U=1-(r['S4'].sum()/MEAS/R)/(3*(r['S2'].sum()/MEAS/R)**2)
    print(T, 'U~',round(U,5), 'absm hist',np.round(r['hist'],3), 'time',round(r['time'],1), flush=True)
