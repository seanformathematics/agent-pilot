import numpy as np, time
N=16; R=1000
def run(T, nsweep, nburn, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    # acceptance prob table for dE = 2*s*nn, nn in {-4..4}, dE in -8..8
    ptab=np.zeros(17)
    for dE in range(-8,9,2) if False else range(-8,9):
        ptab[dE+8]=1.0 if dE<=0 else np.exp(-dE/T)
    m2=np.zeros(R); m4=np.zeros(R); cnt=0
    s=s.astype(np.int8)
    for sw in range(nsweep+nburn):
        for mk in masks:
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE=(2*s*nn).astype(np.int8)
            p=ptab[dE+8]
            acc=(rng.random(size=s.shape,dtype=np.float32)<p)&mk
            s=np.where(acc,-s,s).astype(np.int8)
        if sw>=nburn:
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2+=m*m; m4+=m**4; cnt+=1
    m2/=cnt; m4/=cnt
    return m2,m4
t=time.time()
m2,m4=run(2.27,200,0,1)
print("time per sweep",(time.time()-t)/200)
t=time.time()
m2,m4=run(2.2,1000,0,2)
U=1-m4.mean()/(3*m2.mean()**2)
print("time",time.time()-t,"U(2.2) approx",U, "m2",m2.mean())