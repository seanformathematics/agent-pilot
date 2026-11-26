import numpy as np, time
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
    S2=np.zeros(R); S4=np.zeros(R); cnt=0
    h2=[];h4=[]
    for k in range(nmeas):
        s=sweep(s,T,rng)
        m=s.reshape(R,-1).mean(1)
        m2=m*m
        S2+=m2; S4+=m2*m2
    return S2/nmeas, S4/nmeas
def binder(m2,m4):
    return 1-np.mean(m4)/(3*np.mean(m2)**2)
def jack(m2,m4):
    R=len(m2); U=binder(m2,m4)
    Ui=np.empty(R)
    tot2=m2.sum(); tot4=m4.sum()
    for r in range(R):
        a2=(tot2-m2[r])/(R-1); a4=(tot4-m4[r])/(R-1)
        Ui[r]=1-a4/(3*a2*a2)
    err=np.sqrt((R-1)/R*np.sum((Ui-Ui.mean())**2))
    return U,err
t=time.time()
m2,m4=run(2.3,1000,2000,2000,1)
print("time",time.time()-t)
print("U,err",jack(m2,m4))
