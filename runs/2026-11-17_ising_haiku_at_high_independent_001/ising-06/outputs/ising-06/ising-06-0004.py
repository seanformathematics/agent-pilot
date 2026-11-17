import numpy as np, time, json, os
os.makedirs('/work',exist_ok=True)
N=16
def run(T, R, equil, meas, seed, init=None):
    rng=np.random.default_rng(seed)
    if init is None:
        s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    else:
        s=np.ones((R,N,N),np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    a4=np.exp(-4/T); a8=np.exp(-8/T)
    def sweep(s):
        for p in (0,1):
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            sn=s*nn
            u=rng.random((R,N,N))
            pr=np.where(sn<=0,1.0,np.where(sn==2,a4,a8))
            flip=(u<pr)&masks[p]
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(equil):
        s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); S2h=np.zeros((2,R)); S4h=np.zeros((2,R)); M1=np.zeros(R)
    h=meas//2
    for t in range(meas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1)
        m2=m*m; m4=m2*m2
        S2+=m2; S4+=m4; M1+=np.abs(m)
        k=0 if t<h else 1
        S2h[k]+=m2; S4h[k]+=m4
    return dict(S2=S2,S4=S4,M1=M1,S2h=S2h,S4h=S4h,n=meas)
def ucalc(S2,S4):
    m2=S2.mean(); m4=S4.mean()
    return 1-m4/(3*m2*m2)
def jack(S2,S4):
    R=len(S2)
    U=ucalc(S2,S4)
    Uj=np.empty(R)
    tot2=S2.sum(); tot4=S4.sum()
    for r in range(R):
        Uj[r]=1-((tot4-S4[r])/(R-1))/(3*((tot2-S2[r])/(R-1))**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
t0=time.time()
res=run(2.3,1000,2000,4000,seed=101)
print("time",time.time()-t0)
U,e=jack(res['S2'],res['S4'])
print("U",U,"err",e)
m2h=res['S2h']; m4h=res['S4h']
print("half U:", ucalc(m2h[0],m4h[0]), ucalc(m2h[1],m4h[1]))
print("<m2>",res['S2'].mean()/res['n'],"<|m|>",res['M1'].mean()/res['n'])
