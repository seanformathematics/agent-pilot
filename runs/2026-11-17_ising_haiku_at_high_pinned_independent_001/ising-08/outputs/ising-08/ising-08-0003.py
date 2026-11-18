
import numpy as np, time, json, os
os.makedirs('/work',exist_ok=True)
N=16; R=2000; NEQ=1500; NMEAS=4000
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1]
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def run(T,seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ptab=np.array([min(1.0,np.exp(-2*(k-4)/T)) for k in range(9)],dtype=np.float32)
    def sweep(s):
        for m in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            p=ptab[s*nb+4]
            u=rng.random(s.shape,dtype=np.float32)
            s=np.where((u<p)&m,-s,s)
        return s
    for _ in range(NEQ): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); blk=[]
    bs=0; b2=np.zeros(R); b4=np.zeros(R); BL=500
    for k in range(NMEAS):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m
        S2+=m2; S4+=m2*m2
        b2+=m2; b4+=m2*m2; bs+=1
        if bs==BL:
            blk.append(1-(b4/BL)/(3*(b2/BL)**2)); b2[:]=0;b4[:]=0;bs=0
    return S2/NMEAS, S4/NMEAS, np.array(blk)
res={}
for idx,T in enumerate(Ts):
    t=time.time()
    m2r,m4r,blk=run(T,seed=1000+idx)
    # U from pooled
    M2=m2r.mean(); M4=m4r.mean()
    U=1-M4/(3*M2**2)
    # jackknife over replicas
    jk=[]
    for r in range(R):
        keep=np.ones(R,bool); keep[r]=False
        a2=m2r[keep].mean(); a4=m4r[keep].mean()
        jk.append(1-a4/(3*a2**2))
    jk=np.array(jk)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    res[T]=dict(U=float(U),err=float(err),blocks=blk.tolist())
    print(T, round(U,5), round(err,5), 'block U:',np.round(blk,3), 'time',round(time.time()-t,1), flush=True)
json.dump(res,open('/work/batch1.json','w'))
