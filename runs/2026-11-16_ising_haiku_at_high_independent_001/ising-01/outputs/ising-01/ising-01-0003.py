import numpy as np, time, os
os.makedirs("/work",exist_ok=True)
code = r'''
import numpy as np, sys, time
def run(T, R, n_eq, n_meas, seed, start_ordered):
    N=16
    rng=np.random.default_rng(seed)
    if start_ordered:
        s=np.where(rng.random((R,1,1))<0.5,1,-1).astype(np.int8)*np.ones((R,N,N),np.int8)
    else:
        s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    # acceptance table indexed by delta=s*nn in {-4,-2,0,2,4} -> (delta+4)//2 in 0..4
    # flip prob = min(1, exp(-2*delta/T))
    d=np.array([-4,-2,0,2,4])
    ptab=np.minimum(1.0,np.exp(-2.0*d/T))
    mask_even=((np.arange(N)[:,None]+np.arange(N)[None,:])%2==0)
    masks=[mask_even, ~mask_even]
    def sweep(s):
        for mk in masks:
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            delta=(s*nn).astype(np.int8)
            p=ptab[((delta+4)>>1)]
            acc=(rng.random((R,N,N))<p)&mk
            s=np.where(acc,-s,s).astype(np.int8)
        return s
    for _ in range(n_eq):
        s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); SA=np.zeros(R)
    for _ in range(n_meas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1)
        m2=m*m
        S2+=m2; S4+=m2*m2; SA+=np.abs(m)
    return S2/n_meas, S4/n_meas, SA/n_meas, s
if __name__=="__main__":
    pass
'''
open("/work/ising.py","w").write(code)
import sys; sys.path.insert(0,"/work")
from ising import run
t=time.time()
m2,m4,ma,s=run(2.269,1000,0,50,1,False)
dt=time.time()-t
print("time per sweep (R=1000):",dt/50)
print(m2.mean(), m4.mean())
