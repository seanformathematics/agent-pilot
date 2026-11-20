
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
code = r'''
import numpy as np, sys, time
N=16
T=float(sys.argv[1]); R=int(sys.argv[2]); nsweep=int(sys.argv[3]); ther=int(sys.argv[4]); seed=int(sys.argv[5]); out=sys.argv[6]
rng=np.random.default_rng(seed)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==k) for k in (0,1)]
beta=1.0/T
acc_p=np.exp(-beta*np.arange(-8,9,4)*1.0)  # placeholder
# precompute exp(-beta*dE) for dE=2*s*nn, nn in {-4,-2,0,2,4}, dE in {-8,-4,0,4,8}
pexp={dE:np.exp(-beta*dE) for dE in (4,8)}
def sweep(s):
    for m in masks:
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nn  # energy change if flipped (int8 ok: max 8)
        r=rng.random(s.shape)
        flip=m[None,:,:] & ((dE<=0) | (r < np.where(dE==4,pexp[4],pexp[8]) ))
        s=np.where(flip,-s,s).astype(np.int8)
    return s
t0=time.time()
for _ in range(ther): s=sweep(s)
M=[];M2=[];M4=[];E=[]
meas_every=1
nm=0
for it in range(nsweep):
    s=sweep(s)
    if it%meas_every==0:
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        M2.append(m*m); M4.append(m**4); nm+=1
M2=np.array(M2); M4=np.array(M4)
np.savez(out,m2=M2,m4=M4,T=T,R=R,ther=ther,nsweep=nsweep,seed=seed)
print("done",T,time.time()-t0, nm)
'''
open('/work/ising.py','w').write(code)
import subprocess
r=subprocess.run(['python3','/work/ising.py','2.269','400','200','100','1','/work/test.npz'],capture_output=True,text=True,timeout=300)
print(r.stdout,r.stderr)
