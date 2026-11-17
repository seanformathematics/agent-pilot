import numpy as np, time
N=16
def run(T,R,nequil,nmeas,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    p=np.array([min(1.0,np.exp(-2*(k-4)/T)) for k in range(9)])
    def sweep():
        for mk in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            k=(s*nb+4).astype(np.int8)
            acc=rng.random((R,N,N))<p[k]
            upd=acc&mk
            s[upd]*=-1
    for _ in range(nequil): sweep()
    S2=np.zeros(R); S4=np.zeros(R); n=0
    for _ in range(nmeas):
        sweep()
        m=s.sum(axis=(1,2))/(N*N)
        m2=m*m
        S2+=m2; S4+=m2*m2; n+=1
    return S2/n,S4/n
t=time.time()
a,b=run(2.27,400,500,1000,1)
print("time",time.time()-t)
U=1-np.mean(b)/(3*np.mean(a)**2)
print("U",U)
# jackknife over replicas
R=len(a); J=[]
for r in range(R):
    mask=np.ones(R,bool); mask[r]=False
    J.append(1-np.mean(b[mask])/(3*np.mean(a[mask])**2))
J=np.array(J)
print("jk err", np.sqrt((R-1)/R*np.sum((J-J.mean())**2)))