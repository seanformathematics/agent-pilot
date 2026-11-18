import numpy as np, time, json
N=16
ii,jj=np.indices((N,N)); par=(ii+jj)%2
masks=[par==0, par==1]
def run(T,R,nsweep_eq,nmeas,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    xs=np.arange(-4,5)
    tab=np.minimum(1.0,np.exp(-2.0*xs/T))
    def sweep(s):
        for mk in masks:
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            x=(s*h).astype(np.int8)
            p=tab[x+4]
            flip=(rng.random(s.shape)<p)&mk
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nsweep_eq): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R)
    for _ in range(nmeas):
        s=sweep(s)
        m=s.sum(axis=(1,2),dtype=np.int64)/(N*N)
        m2=m*m
        S2+=m2; S4+=m2*m2
    return S2/nmeas, S4/nmeas
def binder(S2,S4):
    R=len(S2)
    U=1-S4.mean()/(3*S2.mean()**2)
    tot2=S2.sum(); tot4=S4.sum()
    Uj=1-((tot4-S4)/(R-1))/(3*((tot2-S2)/(R-1))**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1]
out={}
t=time.time()
for k,T in enumerate(Ts):
    S2,S4=run(T,1000,1500,1500,100+k)
    U,e=binder(S2,S4)
    out[T]=(U,e)
    print(T,U,e,round(time.time()-t,1),flush=True)
json.dump({str(k):v for k,v in out.items()},open('/work/part1.json','w')) if __import__('os').makedirs('/work',exist_ok=True) is None else None
