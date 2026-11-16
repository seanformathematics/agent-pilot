script = r'''
import numpy as np, sys, time
N=16; R=1000; BURN=1000; SWEEPS=4000
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
seed0=20240601
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
for T in Ts:
    rng=np.random.default_rng(seed0+int(round(T*100)))
    S=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    # acceptance table for s*nb in {-4,-2,0,2,4}
    def half(S,mask):
        nb=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
        dE=2*S*nb  # energy change for flipping
        prob=np.exp(-dE/T)
        u=rng.random(S.shape,dtype=np.float32)
        flip=(u<prob)&mask
        S[flip]*=-1
    s1=s2=s4=np.zeros(R)
    S1=np.zeros(R); S2=np.zeros(R); S4=np.zeros(R); n=0
    t0=time.time()
    for sw in range(BURN+SWEEPS):
        half(S,masks[0]); half(S,masks[1])
        if sw>=BURN:
            m=S.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2=m*m
            S2+=m2; S4+=m2*m2; S1+=np.abs(m); n+=1
    np.savez(f'/work/res_T{T:.2f}.npz',S2=S2/n,S4=S4/n,S1=S1/n,n=n,R=R)
    print(T,time.time()-t0,flush=True)
'''
open('/work/sim.py','w').write(script)
import subprocess
subprocess.Popen('cd /work && nohup python3 sim.py > sim.log 2>&1 &',shell=True)
