
import numpy as np, time, json
N=16;R=128
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
mask=((np.add.outer(np.arange(N),np.arange(N)))%2==0)
def run(T,seed,neq,nm):
    rng=np.random.default_rng(seed)
    s=np.ones((R,N,N),np.int8)
    s[rng.random(R)<0.5]*=-1
    m2=np.zeros(R);m4=np.zeros(R)
    for it in range(neq+nm):
        for c in (mask,~mask):
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb
            acc=rng.random(s.shape)<np.exp(-dE/T)
            s=np.where(acc&c,-s,s)
        if it>=neq:
            m=s.sum((1,2))/N**2
            m2+=m**2;m4+=m**4
    return m2/nm,m4/nm
t=time.time()
out={}
for T in Ts:
    a,b=run(T,int(T*100),1000,6000)
    # jackknife over replicas
    def U(a,b):return 1-b.mean()/(3*a.mean()**2)
    u=U(a,b)
    jk=np.array([U(np.delete(a,i),np.delete(b,i)) for i in range(R)])
    e=np.sqrt((R-1)/R*((jk-jk.mean())**2).sum())
    out[T]=(u,e);print(T,u,e,time.time()-t,flush=True)
json.dump(out,open('/work/o.json','w'))
