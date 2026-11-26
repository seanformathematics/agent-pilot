
import numpy as np, time, json
N=16
def run(T, R, nsweep, nburn, seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    a4=np.exp(-4/T); a8=np.exp(-8/T)
    def half(s,mask):
        h = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE = 2*s*h
        p = np.where(dE<=0,1.0,np.where(dE==4,a4,np.where(dE==8,a8,0.0)))
        flip = (rng.random(s.shape)<p) & mask
        s[flip] *= -1
    m2s=np.zeros((nsweep,R)); m4s=np.zeros((nsweep,R))
    for it in range(nburn+nsweep):
        half(s,masks[0]); half(s,masks[1])
        if it>=nburn:
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2s[it-nburn]=m*m; m4s[it-nburn]=m**4
    return m2s,m4s
res={}
t0=time.time()
for k,T in enumerate([1.9,2.0,2.1,2.2,2.3,2.4]):
    R=512; nsw=10000; nb=3000
    m2,m4=run(T,R,nsw,nb,seed=1000+k)
    M2=m2.mean(0); M4=m4.mean(0)
    Ur=1-M4/(3*M2**2)
    U=1-m4.mean()/(3*m2.mean()**2)
    Uerr=Ur.std(ddof=1)/np.sqrt(R)
    # drift check: first vs second half of production
    h=nsw//2
    U1=1-m4[:h].mean()/(3*m2[:h].mean()**2); U2=1-m4[h:].mean()/(3*m2[h:].mean()**2)
    res[T]=dict(U=U,U_err=Uerr,U_half1=U1,U_half2=U2,n=nsw*R)
    print(T,'U=%.5f err=%.5f halves %.5f %.5f  t=%.1f'%(U,Uerr,U1,U2,time.time()-t0),flush=True)
json.dump(res,open('/work/res_main.json','w'))
