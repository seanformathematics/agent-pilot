import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def make_masks():
    i,j=np.indices((N,N))
    return ((i+j)%2==0), ((i+j)%2==1)
EV,OD=make_masks()
def sweep(s,T,rng,masks):
    for mask in masks:
        nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=2*s*nn
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.maximum(dE,0)/T))
        flip=acc&mask
        s[flip]*=-1
    return s
def run(T,R,nburn,nmeas,seed,start='rand',every=1):
    rng=np.random.default_rng(seed)
    if start=='rand': s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    else: s=np.ones((R,N,N),dtype=np.int8)
    s=s.astype(np.int64)
    masks=(EV,OD)
    for _ in range(nburn): sweep(s,T,rng,masks)
    m1=np.zeros(R); m2=np.zeros(R); m4=np.zeros(R); cnt=0
    series=[]
    for k in range(nmeas):
        sweep(s,T,rng,masks)
        m=s.reshape(R,-1).mean(1)
        series.append(m[:50].copy() if k<2000 else None)
        m2+=m**2; m4+=m**4; cnt+=1
    m2/=cnt; m4/=cnt
    return m2,m4
t=time.time()
m2,m4=run(2.27,500,200,200,1)
print("time per sweep (R=500):",(time.time()-t)/400)
t=time.time()
m2,m4=run(1.5,500,500,500,2,start='ord')
print("T=1.5 U",1-np.mean(m4)/(3*np.mean(m2)**2),"time",time.time()-t)
