
import numpy as np, time, os
N=16
i,j=np.indices((N,N)); MA=((i+j)%2==0); MB=~MA
def run_T(T,R,nburn,nmeas,seed,blk=50):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    keys=np.array([-4,-2,0,2,4]); tab=np.minimum(1.0,np.exp(-2*keys/T))
    def sw(s):
        for M in (MA,MB):
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            x=(s*nn).astype(np.int64)
            p=tab[(x+4)//2]
            flip=(rng.random(s.shape)<p)&M
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nburn): s=sw(s)
    nb=nmeas//blk
    B2=np.zeros((nb,R)); B4=np.zeros((nb,R)); BA=np.zeros((nb,R))
    for b in range(nb):
        acc2=np.zeros(R); acc4=np.zeros(R); accA=np.zeros(R)
        for _ in range(blk):
            s=sw(s)
            m=s.reshape(R,-1).mean(1)
            acc2+=m*m; acc4+=m**4; accA+=np.abs(m)
        B2[b]=acc2/blk; B4[b]=acc4/blk; BA[b]=accA/blk
    return B2,B4,BA
os.makedirs('/work/data',exist_ok=True)
for T in [1.9,2.0]:
    t=time.time()
    B2,B4,BA=run_T(T,512,2000,20000,seed=int(T*1000)+7)
    np.savez(f'/work/data/T{T}.npz',B2=B2,B4=B4,BA=BA)
    m2=B2.mean(); m4=B4.mean()
    print(T,"time",round(time.time()-t,1),"U",1-m4/(3*m2**2),"<|m|>",BA.mean(),
          "first-half U",1-B4[:len(B4)//2].mean()/(3*B2[:len(B2)//2].mean()**2),
          "second-half U",1-B4[len(B4)//2:].mean()/(3*B2[len(B2)//2:].mean()**2))
