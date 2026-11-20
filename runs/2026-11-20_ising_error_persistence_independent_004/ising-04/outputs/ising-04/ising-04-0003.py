import numpy as np, time, json, os
N=16
i,j=np.indices((N,N))
EV=((i+j)%2==0); OD=((i+j)%2==1)
def run(T,R,ntherm,nmeas,seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    acc=np.array([min(1.0,np.exp(-d/T)) for d in range(-8,9)])
    def half(s,mask):
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=(2*s*nn).astype(np.int16)
        flip=(rng.random(s.shape)<acc[dE+8])&mask
        s[flip]*=-1
    for _ in range(ntherm):
        half(s,EV); half(s,OD)
    S2=np.zeros(R); S4=np.zeros(R)
    # also store block-wise sums (10 blocks) per replica for diagnostics
    nb=10; blk2=np.zeros((nb,R)); blk4=np.zeros((nb,R))
    per=nmeas//nb
    for k in range(nmeas):
        half(s,EV); half(s,OD)
        M=s.sum(axis=(1,2),dtype=np.int64)/(N*N)
        m2=M*M; m4=m2*m2
        S2+=m2; S4+=m4
        b=k//per
        if b<nb:
            blk2[b]+=m2; blk4[b]+=m4
    return S2/nmeas, S4/nmeas, blk2/per, blk4/per

def U_from(a2,a4):
    return 1-a4.mean()/(3*a2.mean()**2)

def jack(a2,a4,G=50):
    R=len(a2); idx=np.arange(R)%G
    U0=U_from(a2,a4)
    Us=[]
    for g in range(G):
        keep=idx!=g
        Us.append(U_from(a2[keep],a4[keep]))
    Us=np.array(Us)
    var=(G-1)/G*np.sum((Us-Us.mean())**2)
    return U0, np.sqrt(var)

Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
res={}
t0=time.time()
for n,T in enumerate(Ts):
    a2,a4,b2,b4=run(T,2000,2000,3000,seed=1000+n)
    U,err=jack(a2,a4)
    # consistency: halves of time series
    Uh1=1-(b4[:5].mean())/(3*b2[:5].mean()**2); Uh2=1-(b4[5:].mean())/(3*b2[5:].mean()**2)
    res[T]=dict(U=float(U),err=float(err),seed=1000+n,h1=float(Uh1),h2=float(Uh2))
    print(T,res[T],"elapsed",round(time.time()-t0,1),flush=True)
    json.dump(res,open('/work/run1.json','w'),indent=1)
