
import numpy as np, time, os
os.makedirs("/work",exist_ok=True)
N=16
def run(T, R, nequil, nmeas, every, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    # acceptance for x = s*nn in {-4,-2,0,2,4}; dE=2*x
    accT={x: (1.0 if -2*x<=0 else np.exp(-2*x/T)) for x in [-4,-2,0,2,4]}
    ptab=np.zeros(9,dtype=np.float32)
    for x in [-4,-2,0,2,4]: ptab[x+4]=accT[x]
    def sweep(s):
        for m in masks:
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            x=(s*nn).astype(np.int16)+4
            p=ptab[x]
            flip=m[None]&(rng.random((R,N,N),dtype=np.float32)<p)
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nequil): s=sweep(s)
    M2=np.zeros(R); M4=np.zeros(R); cnt=0
    for k in range(nmeas):
        for _ in range(every): s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m
        M2+=m2; M4+=m2*m2; cnt+=1
    return M2/cnt, M4/cnt
t=time.time()
M2,M4=run(2.269,500,50,20,1,1)
print("time",time.time()-t)
