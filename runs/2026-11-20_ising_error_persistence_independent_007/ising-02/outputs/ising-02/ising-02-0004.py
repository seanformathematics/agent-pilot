
import os,glob,subprocess
for f in glob.glob('/work/*.npz')+glob.glob('/tmp/*.npz'):
    try: os.remove(f)
    except Exception as e: print(e)
print(subprocess.run(['df','-h','/work','/tmp'],capture_output=True,text=True).stdout)
code = r'''
import numpy as np, sys, time
N=16
T=float(sys.argv[1]); R=int(sys.argv[2]); nsweep=int(sys.argv[3]); ther=int(sys.argv[4]); seed=int(sys.argv[5]); out=sys.argv[6]
BS=50
rng=np.random.default_rng(seed)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==k) for k in (0,1)]
beta=1.0/T
p4=np.exp(-4*beta); p8=np.exp(-8*beta)
def sweep(s):
    for m in masks:
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nn
        r=rng.random(s.shape)
        acc=(dE<=0)|((dE==4)&(r<p4))|((dE==8)&(r<p8))
        flip=m[None,:,:]&acc
        s=np.where(flip,-s,s)
    return s
t0=time.time()
for _ in range(ther): s=sweep(s)
B2=[];B4=[];BE=[]
acc2=np.zeros(R);acc4=np.zeros(R);cnt=0
for it in range(nsweep):
    s=sweep(s)
    m=s.reshape(R,-1).mean(axis=1)
    acc2+=m*m; acc4+=m**4; cnt+=1
    if cnt==BS:
        B2.append(acc2/BS); B4.append(acc4/BS); acc2=np.zeros(R); acc4=np.zeros(R); cnt=0
np.savez(out,m2=np.array(B2),m4=np.array(B4),T=T,R=R,ther=ther,nsweep=nsweep,seed=seed,BS=BS)
print("done",T,round(time.time()-t0,1))
'''
open('/work/ising.py','w').write(code)
