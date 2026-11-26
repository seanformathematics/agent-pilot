import numpy as np, time, json
N=16
i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
A=((i+j)%2==0); B=~A
def sweep(s,T,rng):
    for mask in (A,B):
        h = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        p = np.exp(-(2*s*h).astype(np.float64)/T)
        r = rng.random(s.shape)
        s = np.where(mask & (r<p), -s, s).astype(np.int8)
    return s
def run(T,R,neq,nmeas,seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    for k in range(neq): s=sweep(s,T,rng)
    S2=np.zeros((2,R)); S4=np.zeros((2,R))
    for k in range(nmeas):
        s=sweep(s,T,rng)
        m=s.reshape(R,-1).mean(1); m2=m*m
        half=0 if k<nmeas//2 else 1
        S2[half]+=m2; S4[half]+=m2*m2
    h=nmeas//2
    return S2/h, S4/h
def jack(m2,m4):
    R=len(m2); U=1-m4.mean()/(3*m2.mean()**2)
    tot2=m2.sum(); tot4=m4.sum()
    a2=(tot2-m2)/(R-1); a4=(tot4-m4)/(R-1)
    Ui=1-a4/(3*a2*a2)
    return U, np.sqrt((R-1)/R*np.sum((Ui-Ui.mean())**2))
res={}
for T,seed in [(2.2,104),(2.3,105),(2.4,106)]:
    t=time.time()
    S2,S4=run(T,1000,3000,3000,seed)
    m2=S2.mean(0); m4=S4.mean(0)
    U,e=jack(m2,m4)
    Uh=[jack(S2[k],S4[k])[0] for k in range(2)]
    res[T]=dict(U=U,err=e,U_half=Uh,seed=seed)
    print(T,U,e,"halves",Uh,"time",round(time.time()-t,1),flush=True)
json.dump(res,open('/work/res_b.json','w'))
