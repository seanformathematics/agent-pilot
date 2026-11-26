
import numpy as np, time
N=16; R=512
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def run(T, seed, NEQ=3000, NMEAS=20000, BLK=200):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep(s):
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE=2*s*nb
            acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.maximum(dE,0)/T))
            s[acc&m]*=-1
    for k in range(NEQ): sweep(s)
    nbk=NMEAS//BLK
    m2b=np.zeros((nbk,R)); m4b=np.zeros((nbk,R))
    for k in range(NMEAS):
        sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        b=k//BLK
        m2b[b]+=m*m; m4b[b]+=m**4
    m2b/=BLK; m4b/=BLK
    np.savez(f"/work/T{T:.2f}.npz",m2b=m2b,m4b=m4b)
    return m2b,m4b
t=time.time()
for T,seed in [(2.0,1200),(2.1,1210)]:
    run(T,seed); print("done",T,time.time()-t)
def binder_jk(m2b,m4b):
    m2=m2b.mean(0); m4=m4b.mean(0)  # per replica
    def U(a2,a4): return 1-a4.mean()/(3*a2.mean()**2)
    U0=U(m2,m4)
    R=len(m2)
    jk=np.array([U(np.delete(m2,i),np.delete(m4,i)) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return U0,err
d=np.load("/work/T1.90.npz")
print("T1.9",binder_jk(d['m2b'],d['m4b']))
# check equilibration: first half vs second half
m2b,m4b=d['m2b'],d['m4b']
print("first half",binder_jk(m2b[:50],m4b[:50]),"second half",binder_jk(m2b[50:],m4b[50:]))
