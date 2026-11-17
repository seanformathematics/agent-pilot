
import numpy as np, time, json
N=16; R=1000
def run(T, R, nburn, nmeas, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    acc=np.array([1.0 if 2*(x-4)<=0 else np.exp(-2*(x-4)/T) for x in range(9)])
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    def sweep():
        for p in (0,1):
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            sn=(s*nn).astype(np.int16)+4
            a=acc[sn]
            u=rng.random((R,N,N))
            flip=masks[p][None]&(u<a)
            s[flip]*=-1
    for _ in range(nburn): sweep()
    S2=np.zeros(R); S4=np.zeros(R); cnt=0
    for _ in range(nmeas):
        sweep()
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m
        S2+=m2; S4+=m2*m2; cnt+=1
    return S2/cnt, S4/cnt
def U_and_err(A2,A4):
    R=len(A2)
    u=1-A4.mean()/(3*A2.mean()**2)
    tot2=A2.sum(); tot4=A4.sum()
    jk=np.array([1-((tot4-A4[k])/(R-1))/(3*((tot2-A2[k])/(R-1))**2) for k in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return u,err
res={}
for i,T in enumerate([2.2,2.3,2.4,2.5,2.6,2.7,2.8]):
    t=time.time()
    A2,A4=run(T,R,2000,3000,200+i)
    u,e=U_and_err(A2,A4)
    res[T]=dict(U=u,err=e,m2=A2.mean(),seed=200+i)
    print(T,u,e,A2.mean(),time.time()-t,flush=True)
json.dump(res,open('/work/part2.json','w'))
