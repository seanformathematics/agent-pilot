
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def make_masks(N):
    i,j=np.indices((N,N))
    return [( (i+j)%2==p) for p in (0,1)]
MASKS=make_masks(N)
def run(T,R,nsweep,neq,seed,S=None):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    # acceptance table for s*nn in {-4,-2,0,2,4}: flip dE=2*s*nn
    acc={x:min(1.0,np.exp(-2*x/T)) for x in (-4,-2,0,2,4)}
    tab=np.zeros(9)
    for x in (-4,-2,0,2,4): tab[x+4]=acc[x]
    # per replica accumulators
    s2=np.zeros(R); s4=np.zeros(R); cnt=0
    for sweep in range(neq+nsweep):
        for p in (0,1):
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            prod=(s*nn).astype(np.int16)  # in {-4..4}
            pr=tab[prod+4]
            u=rng.random((R,N,N))
            flip=(u<pr)&MASKS[p]
            s=np.where(flip,-s,s).astype(np.int8)
        if sweep>=neq:
            m=s.reshape(R,-1).mean(axis=1)
            m2=m*m
            s2+=m2; s4+=m2*m2; cnt+=1
    return s2/cnt, s4/cnt
t=time.time()
a,b=run(2.269,1000,50,10,1)
print('time per sweep R=1000',(time.time()-t)/60)
print(a.mean(), b.mean())
