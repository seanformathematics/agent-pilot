
script = r'''
import numpy as np, time, sys
N=16
T=float(sys.argv[1]); seed=int(sys.argv[2]); R=int(sys.argv[3]); burn=int(sys.argv[4]); nmeas=int(sys.argv[5])
rng=np.random.default_rng(seed)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
acc=np.array([min(1.0,np.exp(-d/T)) for d in range(-8,9)],dtype=np.float32)
def sweep():
    for m in masks:
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=(2*s*nn).astype(np.int8)
        r=rng.random(s.shape,dtype=np.float32)
        flip=(r<acc[dE+8]) & m
        s[flip]*=-1
t0=time.time()
for k in range(burn): sweep()
S2=np.zeros(R); S4=np.zeros(R); S1=np.zeros(R)
blk=[]; bm2=np.zeros(R); bm4=np.zeros(R); bn=0
blocksize=max(1,nmeas//20)
for k in range(nmeas):
    sweep()
    m=s.sum(axis=(1,2),dtype=np.int64)/(N*N)
    m2=m*m
    S2+=m2; S4+=m2*m2; S1+=np.abs(m)
    bm2+=m2; bm4+=m2*m2; bn+=1
    if bn==blocksize:
        blk.append(1-(bm4/bn).mean()/(3*((bm2/bn).mean())**2))
        bm2[:]=0; bm4[:]=0; bn=0
np.savez(f'/work/res_T{T}_s{seed}.npz',S2=S2,S4=S4,S1=S1,nmeas=nmeas,blk=np.array(blk),T=T,seed=seed,R=R,burn=burn)
print('done',T,seed,'time',time.time()-t0)
'''
open('/work/sim.py','w').write(script)
import subprocess
out=subprocess.run(['python','/work/sim.py','2.3','101','2000','3000','6000'],capture_output=True,text=True,timeout=550)
print(out.stdout,out.stderr[-2000:])
import numpy as np
d=np.load('/work/res_T2.3_s101.npz')
print('blocks U:',np.round(d['blk'],4))
m2=d['S2']/d['nmeas']; m4=d['S4']/d['nmeas']
U=1-m4.mean()/(3*m2.mean()**2)
print('U pooled',U)
