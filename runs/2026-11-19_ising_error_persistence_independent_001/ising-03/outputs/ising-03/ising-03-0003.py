
import numpy as np, time, json
N=16
temps=[1.9,2.0,2.1,2.2,2.3,2.4]
def run(T, R, nequil, nmeas, seed, nblocks=6):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    def sweep():
        for m in masks:
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*(s*h)
            p=np.where(dE<=0,1.0,np.exp(-dE/T))
            flip=(rng.random(s.shape)<p)&m
            s[flip]*=-1
    for _ in range(nequil): sweep()
    per=nmeas//nblocks
    B2=[];B4=[]
    for b in range(nblocks):
        S2=np.zeros(R); S4=np.zeros(R)
        for _ in range(per):
            sweep()
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2=m*m; S2+=m2; S4+=m2*m2
        B2.append(S2/per); B4.append(S4/per)
    return np.array(B2), np.array(B4)  # shape (nblocks,R)
def Ufun(a2,a4):
    x2=a2.mean(); x4=a4.mean()
    return 1-x4/(3*x2*x2)
def binder_jk(a2,a4):
    R=len(a2)
    full=Ufun(a2,a4)
    S2=a2.sum(); S4=a4.sum()
    # leave-one-out via sums (fast)
    x2=(S2-a2)/(R-1); x4=(S4-a4)/(R-1)
    jk=1-x4/(3*x2*x2)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return full,err
t0=time.time()
out={}
for k,T in enumerate(temps):
    B2,B4=run(T,R=1000,nequil=1500,nmeas=3000,seed=20240+k)
    a2=B2.mean(0); a4=B4.mean(0)
    u,e=binder_jk(a2,a4)
    blockU=[Ufun(B2[b],B4[b]) for b in range(B2.shape[0])]
    out[T]=dict(U=u,err=e,blocksU=[round(x,5) for x in blockU])
    print(T,round(u,5),round(e,5),'blocks',np.round(blockU,4),round(time.time()-t0,1),flush=True)
json.dump({str(k):v for k,v in out.items()},open('/work/prod1.json','w'),default=float)
