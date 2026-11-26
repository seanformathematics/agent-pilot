
import subprocess, pickle, numpy as np, os
script = r'''
import numpy as np, time, os, sys, json
N=16; R=1000
i,j=np.indices((N,N)); MA=((i+j)%2==0); MB=~MA
T=float(sys.argv[1]); seed=int(sys.argv[2]); disc=int(sys.argv[3]); nmeas=int(sys.argv[4])
beta=1.0/T
rng=np.random.default_rng(seed)
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
blk=500; blocks=[]
S2=np.zeros(R); S4=np.zeros(R); cnt=0
bS2=np.zeros(R); bS4=np.zeros(R); bn=0
tot=disc+nmeas
for k in range(tot):
    for mask in (MA,MB):
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nb
        p=np.where(dE<=0,1.0,np.exp(-beta*np.maximum(dE,0)))
        flip=(rng.random(s.shape)<p)&mask
        s[flip]*=-1
    if k>=disc:
        M=s.reshape(R,-1).sum(1).astype(np.float64)
        S2+=M*M; S4+=M**4; cnt+=1
        bS2+=M*M; bS4+=M**4; bn+=1
        if bn==blk:
            blocks.append([float(bS2.mean()/blk/N**4), float(bS4.mean()/blk/N**8)])
            bS2[:]=0; bS4[:]=0; bn=0
    if k<disc and (k+1)%blk==0:
        pass
np.savez(f"/work/res_T{T:.3f}_s{seed}.npz", S2=S2/cnt, S4=S4/cnt, blocks=np.array(blocks), disc=disc, nmeas=nmeas)
'''
open('/work/prod.py','w').write(script)
import time
out=[]
for T,seed in [(1.9,101),(2.0,102),(2.1,103)]:
    t=time.time()
    r=subprocess.run(["python","/work/prod.py",str(T),str(seed),"3000","4000"],capture_output=True,text=True,cwd="/work",timeout=400)
    out.append((T,r.returncode,r.stderr[-500:],round(time.time()-t,1)))
print(out)
