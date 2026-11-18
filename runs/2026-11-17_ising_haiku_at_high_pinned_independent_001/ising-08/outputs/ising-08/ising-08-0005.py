
import numpy as np, time, json
N=16; R=2000; NEQ=1500; NMEAS=4000
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
    S2=np.zeros(R); S4=np.zeros(R); H=[np.zeros(R),np.zeros(R)]
    for k in range(NMEAS):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m; S2+=m2; S4+=m2*m2
        H[0 if k<NMEAS//2 else 1]+=m2
    return S2/NMEAS,S4/NMEAS,H[0]/(NMEAS//2),H[1]/(NMEAS//2)
out={}
for idx,T in enumerate([2.2,2.3,2.4,2.5]):
    t=time.time()
    m2r,m4r,h1,h2=run(T,seed=2000+idx)
    U=1-m4r.mean()/(3*m2r.mean()**2)
    jk=np.array([1-(m4r.sum()-m4r[r])/(R-1)/(3*((m2r.sum()-m2r[r])/(R-1))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    out[T]=dict(U=float(U),err=float(err))
    print(T,'U',round(U,6),'err',float(err),'m2 1st half',round(h1.mean(),5),'2nd half',round(h2.mean(),5),'time',round(time.time()-t,1),flush=True)
json.dump(out,open('/work/batch2.json','w'))
