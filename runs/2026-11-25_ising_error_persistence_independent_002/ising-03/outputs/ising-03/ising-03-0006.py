
import numpy as np, time, json
N=16
i,j=np.indices((N,N))
MA=((i+j)%2==0); MB=~MA

def run(T, R, nsweep, nburn, seed, blocks=10):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int16)
    acc=np.array([1.0 if d<=0 else np.exp(-d/T) for d in range(-8,9)])
    def half(s,mask):
        nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        p=acc[2*s*nb+8]
        flip=(rng.random(s.shape)<p)&mask
        s[flip]*=-1
    for t in range(nburn):
        half(s,MA); half(s,MB)
    S2=np.zeros(R); S4=np.zeros(R)
    bs=nsweep//blocks; blU=[]
    for b in range(blocks):
        b2=np.zeros(R); b4=np.zeros(R)
        for t in range(bs):
            half(s,MA); half(s,MB)
            m=s.reshape(R,-1).mean(axis=1); m2=m*m
            b2+=m2; b4+=m2*m2
        S2+=b2; S4+=b4
        a=b2.mean()/bs; bb=b4.mean()/bs
        blU.append(1-bb/(3*a*a))
    return S2,S4,nsweep,np.array(blU)

def binder(S2,S4,cnt):
    R=len(S2)
    def U_of(idx):
        a=S2[idx].mean()/cnt; b=S4[idx].mean()/cnt
        return 1-b/(3*a*a)
    U=U_of(np.arange(R))
    jk=np.array([U_of(np.delete(np.arange(R),k)) for k in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return U,err

Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
t0=time.time()
for k,T in enumerate(Ts):
    S2,S4,c,bU=run(T,512,4000,6000,5000+k)
    U,err=binder(S2,S4,c)
    print(f"T={T} U={U:.5f} err={err:.5f} blockU_mean={bU.mean():.5f} blockU_sd={bU.std(ddof=1):.5f} t={time.time()-t0:.0f}s",flush=True)
