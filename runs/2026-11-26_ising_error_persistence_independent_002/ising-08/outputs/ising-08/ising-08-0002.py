
import numpy as np, time, json
N=16
def run(R, T, ntherm, nmeas, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    def half(s,mask):
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nb
        r=rng.random(s.shape)
        pr=np.exp(-np.clip(dE,0,None)/T)
        flip=(r<pr)&mask
        s[flip]*=-1
    for k in range(ntherm):
        half(s,masks[0]); half(s,masks[1])
    m2=np.zeros(R); m4=np.zeros(R)
    for k in range(nmeas):
        half(s,masks[0]); half(s,masks[1])
        m=s.reshape(R,-1).mean(1,dtype=np.float64)
        m2+=m**2; m4+=m**4
    return m2/nmeas, m4/nmeas
def ujack(m2,m4):
    R=len(m2)
    def U(a2,a4): return 1-np.mean(a4)/(3*np.mean(a2)**2)
    full=U(m2,m4)
    loo=np.array([U(np.delete(m2,i),np.delete(m4,i)) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((loo-loo.mean())**2))
    return full,err
out={}
for T in [1.9,2.0]:
    t=time.time()
    seed=int(T*1000)
    m2,m4=run(1000,T,3000,20000,seed)
    U,e=ujack(m2,m4)
    out[T]=dict(U=U,err=e,seed=seed)
    print(T,U,e,time.time()-t,flush=True)
json.dump(out,open('/work/res_a.json','w'))
