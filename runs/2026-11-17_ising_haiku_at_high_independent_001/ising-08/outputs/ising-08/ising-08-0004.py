
import numpy as np, time, json, os
os.makedirs('/work',exist_ok=True)
N=16
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
maskA=((ii+jj)%2==0)[None]; maskB=~maskA
def run(T, R, nburn, nmeas, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    p2=np.exp(-4/T); p4=np.exp(-8/T)
    def sweep(s):
        for mask in (maskA,maskB):
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE2=s*nn
            r=rng.random(s.shape)
            flip=mask & ((dE2<=0) | ((dE2==2)&(r<p2)) | ((dE2==4)&(r<p4)))
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nburn): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); E1=np.zeros(R)
    for _ in range(nmeas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(1,dtype=np.float64)
        m2=m*m; S2+=m2; S4+=m2*m2
    return S2/nmeas, S4/nmeas
def U_est(m2,m4): return 1-np.mean(m4)/(3*np.mean(m2)**2)
def jack(m2,m4,B=100):
    R=len(m2); U=U_est(m2,m4)
    blocks=np.array_split(np.arange(R),B)
    Us=np.array([U_est(np.delete(m2,b),np.delete(m4,b)) for b in blocks])
    var=(B-1)/B*np.sum((Us-Us.mean())**2)
    return U, np.sqrt(var)
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1]
for k,T in enumerate(Ts):
    t=time.time()
    m2,m4=run(T,4000,1000,3000,1000+k)
    U,e=jack(m2,m4)
    d={'T':T,'U':U,'err':e,'m2':float(m2.mean())}
    json.dump(d,open(f'/work/res_{T}.json','w'))
    print(d, round(time.time()-t,1), flush=True)
