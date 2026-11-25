import numpy as np, time, os
os.makedirs('/work', exist_ok=True)
code = r'''
import numpy as np, sys, time, json
N=16
T=float(sys.argv[1]); R=int(sys.argv[2]); nsweep_eq=int(sys.argv[3]); nmeas=int(sys.argv[4]); seed=int(sys.argv[5])
rng=np.random.Generator(np.random.PCG64(seed))
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
# acceptance table indexed by s*nn in {-4,-2,0,2,4}: dE = 2*s*nn
acc={}
for x in (-4,-2,0,2,4):
    dE=2*x
    acc[x]=1.0 if dE<=0 else np.exp(-dE/T)
tab=np.zeros(9)
for x in (-4,-2,0,2,4): tab[x+4]=acc[x]
mask=np.zeros((N,N),bool)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mask0=((ii+jj)%2==0); mask1=~mask0
def sweep():
    for m in (mask0,mask1):
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int16)
        x=(s*nn).astype(np.int16)  # in {-4..4}
        p=tab[x+4]
        u=rng.random((R,N,N))
        flip=(u<p)&m
        s[flip]*=-1
for _ in range(nsweep_eq): sweep()
S2=np.zeros(R); S4=np.zeros(R); S2sq=np.zeros(R)
t0=time.time()
for k in range(nmeas):
    sweep()
    M=s.reshape(R,-1).sum(1).astype(np.float64)/(N*N)
    m2=M*M
    S2+=m2; S4+=m2*m2
np.savez(f'/work/res_T{T:.3f}_s{seed}.npz',S2=S2,S4=S4,R=R,nmeas=nmeas,T=T,seed=seed)
print(T,seed,'time',time.time()-t0, 'U_naive', 1-(S4.sum()/nmeas/R)/(3*(S2.sum()/nmeas/R)**2))
'''
open('/work/sim.py','w').write(code)
import subprocess
t=time.time()
r=subprocess.run(['python','/work/sim.py','2.269','500','200','200','1'],capture_output=True,text=True)
print(r.stdout,r.stderr,time.time()-t)
