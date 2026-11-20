
import numpy as np, time, os, sys
os.makedirs('/work',exist_ok=True)
N=16; R=1000
T=float(sys.argv[1]) if len(sys.argv)>1 else 1.9
EQ=3000; MEAS=30000; B=50; seed=int(1000+round(T*100))
rng=np.random.default_rng(seed)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def sweep(s):
    for p in (0,1):
        nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=2*s*nn
        pr=np.exp(-np.clip(dE,0,None)/T).astype(np.float32)
        acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<pr)
        flip=acc&masks[p]
        s[flip]*=-1
    return s
t0=time.time()
for _ in range(EQ): sweep(s)
nb=MEAS//B
bm2=np.zeros((nb,R)); bm4=np.zeros((nb,R)); bm1=np.zeros((nb,R))
for b in range(nb):
    a2=np.zeros(R); a4=np.zeros(R); a1=np.zeros(R)
    for k in range(B):
        sweep(s)
        m=s.reshape(R,-1).mean(1)
        a1+=np.abs(m); a2+=m*m; a4+=m**4
    bm1[b]=a1/B; bm2[b]=a2/B; bm4[b]=a4/B
np.savez(f'/work/T{T:.3f}.npz',bm1=bm1,bm2=bm2,bm4=bm4,T=T,EQ=EQ,B=B)
print("done T",T,"time",time.time()-t0, "mean m2", bm2.mean())
