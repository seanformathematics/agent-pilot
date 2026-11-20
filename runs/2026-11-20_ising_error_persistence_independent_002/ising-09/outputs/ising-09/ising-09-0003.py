import numpy as np, time, json
N=16
i,j=np.indices((N,N))
MA=((i+j)%2==0); MB=~MA

def run(T, R, n_eq, n_meas, seed, blk=250):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    acc={4:np.exp(-4/T),8:np.exp(-8/T)}
    def half(s,mask):
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nn
        p=np.where(dE<=0,1.0,np.where(dE==4,acc[4],np.where(dE==8,acc[8],0.0)))
        flip=(rng.random((R,N,N))<p)&mask
        s[flip]*=-1
    tot=n_eq+n_meas
    S2=np.zeros(R); S4=np.zeros(R); cnt=0
    blocks=[]
    b2=np.zeros(R); b4=np.zeros(R); bc=0
    for sw in range(tot):
        half(s,MA); half(s,MB)
        if sw>=n_eq:
            m=s.reshape(R,-1).mean(axis=1); m2=m*m
            S2+=m2; S4+=m2*m2; cnt+=1
            b2+=m2; b4+=m2*m2; bc+=1
            if bc==blk:
                blocks.append(1-np.mean(b4/bc)/(3*np.mean(b2/bc)**2))
                b2[:]=0; b4[:]=0; bc=0
    return S2/cnt, S4/cnt, np.array(blocks)

def binder_jk(S2,S4,G=50):
    # delete-group jackknife over replicas
    R=len(S2)
    idx=np.arange(R)%G
    def U(a2,a4): return 1-a4.mean()/(3*a2.mean()**2)
    full=U(S2,S4)
    jk=[]
    for g in range(G):
        k=idx!=g
        jk.append(U(S2[k],S4[k]))
    jk=np.array(jk)
    err=np.sqrt((G-1)/G*np.sum((jk-jk.mean())**2))
    return full,err

out={}
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    t=time.time()
    S2,S4,bl=run(T,2000,1000,3000,seed=int(T*1000)+7)
    U,err=binder_jk(S2,S4)
    out[T]=dict(U=U,err=err,n=2000,blocks=bl.round(4).tolist())
    print(f"T={T} U={U:.5f} err={err:.5f} time={time.time()-t:.1f}s")
    print("  block U:",np.round(bl,4))
json.dump({str(k):v for k,v in out.items()},open("/work/run1.json","w"))
