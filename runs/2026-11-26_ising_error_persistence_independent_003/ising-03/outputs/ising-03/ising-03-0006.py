import numpy as np, time, os, json
N=16
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
def sweep(s,T,rng):
    for c in (0,1):
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*h
        p=np.exp(-np.clip(dE,0,None)/T)
        acc=(dE<=0)|(rng.random(s.shape)<p)
        s=np.where(acc&masks[c],-s,s).astype(np.int8)
    return s
R=1000
res={}
for T,seed in [(2.2,104),(2.3,105),(2.4,106)]:
    t0=time.time()
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    for _ in range(2000): s=sweep(s,T,rng)
    S2=np.zeros(R); S4=np.zeros(R); n=0
    blocks=[]
    b2=np.zeros(R); b4=np.zeros(R); nbk=0
    for k in range(10000):
        s=sweep(s,T,rng)
        if k%5==0:
            m=s.reshape(R,-1).mean(1,dtype=np.float64)
            S2+=m**2; S4+=m**4; n+=1
            b2+=m**2; b4+=m**4; nbk+=1
            if nbk==200:
                blocks.append(1-(b4.mean()/nbk)/(3*(b2.mean()/nbk)**2)); b2=np.zeros(R); b4=np.zeros(R); nbk=0
    S2/=n; S4/=n
    U=1-S4.mean()/(3*S2.mean()**2)
    tot2=S2.sum(); tot4=S4.sum()
    Uj=np.array([1-((tot4-S4[i])/(R-1))/(3*((tot2-S2[i])/(R-1))**2) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    res[T]=dict(U=float(U),err=float(err),n=n*R,blocks=[float(x) for x in blocks])
    print(T, "U=",U,"err=",err,"blocks=",np.round(blocks,4),"time",time.time()-t0, flush=True)
    with open(f'/work/res_{T}.json','w') as f: json.dump(res[T],f)