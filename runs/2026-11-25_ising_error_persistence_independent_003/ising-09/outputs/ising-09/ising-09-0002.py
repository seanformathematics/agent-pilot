
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def run(T, R, nsweep, nburn, seed, outname):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    acc={d:np.exp(-d/T) for d in (4,8)}
    S2=np.zeros(R); S4=np.zeros(R); cnt=0
    # per-sweep measurements also kept in block form for error checks
    blocks2=[]; blocks4=[]
    bs=0; b2=np.zeros(R); b4=np.zeros(R)
    for sw in range(nsweep):
        for p in (0,1):
            nb=(np.roll(s,1,1).astype(np.int16)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nb  # energy change for flipping (J=1): dE=2 s * sum nb
            prob=np.zeros(s.shape)
            prob[dE==4]=acc[4]; prob[dE==8]=acc[8]
            prob[dE<=0]=1.0
            u=rng.random(s.shape)
            flip=(u<prob)&masks[p]
            s=np.where(flip,-s,s).astype(np.int8)
        if sw>=nburn:
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2=m*m
            S2+=m2; S4+=m2*m2; cnt+=1
            b2+=m2; b4+=m2*m2; bs+=1
            if bs==100:
                blocks2.append(b2/bs); blocks4.append(b4/bs); b2=np.zeros(R); b4=np.zeros(R); bs=0
    return S2/cnt, S4/cnt, np.array(blocks2), np.array(blocks4)

if __name__=="__main__":
    t=time.time()
    m2,m4,b2,b4=run(2.3,500,300,100,123,None)
    print("time",time.time()-t, "per sweep", (time.time()-t)/300)
    U=1-m4.mean()/(3*m2.mean()**2)
    print("U",U, m2.mean())
