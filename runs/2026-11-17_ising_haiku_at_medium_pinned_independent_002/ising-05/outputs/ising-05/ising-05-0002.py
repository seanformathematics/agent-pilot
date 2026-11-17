
import numpy as np, time, json, os
os.makedirs('/work',exist_ok=True)
N=16; R=400; NTHERM=600; NMEAS=2500
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
def run(T,seed):
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    # acceptance probs for dE=2*s*nn, s*nn in {-4..4}
    pacc={d:np.exp(-2*d/T) for d in (2,4,6,8)}
    def sweep():
        for c in range(2):
            nn=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
            sn=S*nn  # in {-4,-2,0,2,4}
            # dE = 2*sn ; accept if dE<=0 or rand<exp(-dE/T)
            pr=np.ones(S.shape)
            for d,p in pacc.items():
                pr=np.where(sn==d//2*0+ (d//2) if False else sn==d//2,p,pr)
            u=rng.random(S.shape)
            flip=masks[c][None]&(u<pr)
            S[flip]*=-1
    # pr: sn=2 -> dE=4 ; sn=4 -> dE=8
    # redefine mapping explicitly
    pa={2:np.exp(-4/T),4:np.exp(-8/T)}
    def sweep2():
        for c in range(2):
            nn=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
            sn=S*nn
            pr=np.where(sn==2,pa[2],np.where(sn==4,pa[4],1.0))
            u=rng.random(S.shape)
            flip=masks[c][None]&(u<pr)
            S[flip]=-S[flip]
    for _ in range(NTHERM): sweep2()
    m2=np.zeros(R); m4=np.zeros(R); cnt=0
    M2=[];M4=[]
    for _ in range(NMEAS):
        sweep2()
        m=S.sum(axis=(1,2))/(N*N)
        M2.append(m*m); M4.append(m**4)
    M2=np.array(M2); M4=np.array(M4)  # (NMEAS,R)
    return M2,M4
def binder_from(a2,a4):
    # a2,a4: per-replica time means arrays
    return 1-a4.mean()/(3*a2.mean()**2)
def jack(M2,M4):
    a2=M2.mean(0); a4=M4.mean(0)
    Rn=len(a2)
    U=binder_from(a2,a4)
    Uj=np.empty(Rn)
    tot2=a2.sum(); tot4=a4.sum()
    for k in range(Rn):
        s2=(tot2-a2[k])/(Rn-1); s4=(tot4-a4[k])/(Rn-1)
        Uj[k]=1-s4/(3*s2*s2)
    err=np.sqrt((Rn-1)/Rn*np.sum((Uj-Uj.mean())**2))
    return U,err
res={}
t0=time.time()
for idx,T in enumerate(Ts):
    M2,M4=run(T,1000+idx)
    U,err=jack(M2,M4)
    res[T]=(U,err)
    print(T,round(U,5),round(err,5),round(time.time()-t0,1),flush=True)
    json.dump({str(k):v for k,v in res.items()},open('/work/res1.json','w'))
    if time.time()-t0>480: break
