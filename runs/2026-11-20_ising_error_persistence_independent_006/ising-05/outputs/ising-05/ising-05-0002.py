
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def make_masks(N):
    i,j=np.indices((N,N))
    return ((i+j)%2==0), ((i+j)%2==1)
MA,MB=make_masks(N)
def sweep(s,T,rng,acc4,acc8):
    # s shape (R,N,N) int8; acc table by sum: prob for s*nn
    for M in (MA,MB):
        nn = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        # dE = 2*s*nn ; accept if exp(-dE/T)
        x = (s*nn)  # in {-4,-2,0,2,4}
        p = np.where(x<=0, np.where(x==0,acc0,acc4), 1.0)  # placeholder replaced below
        r = rng.random(s.shape)
        flip = (r < p) & M
        s = np.where(flip, -s, s)
    return s
def run_T(T,R,nburn,nmeas,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    # acceptance probs for x=s*nn in {-4,-2,0,2,4}: dE=2x
    pa={x:min(1.0,np.exp(-2*x/T)) for x in (-4,-2,0,2,4)}
    tab=np.zeros(5); keys=[-4,-2,0,2,4]
    for k,x in enumerate(keys): tab[k]=pa[x]
    def sw(s):
        for M in (MA,MB):
            nn = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            x=(s*nn).astype(np.int64)
            p=tab[(x+4)//2]
            flip=(rng.random(s.shape)<p)&M
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nburn): s=sw(s)
    m2=[];m4=[];am=[]
    for _ in range(nmeas):
        s=sw(s)
        m=s.reshape(R,-1).mean(1)
        m2.append(m*m); m4.append(m**4); am.append(np.abs(m))
    return np.array(m2),np.array(m4),np.array(am)
t=time.time()
m2,m4,am=run_T(2.269,256,200,200,1)
print("time",time.time()-t)
print("U est", 1-m4.mean()/(3*m2.mean()**2), "<|m|>",am.mean())
