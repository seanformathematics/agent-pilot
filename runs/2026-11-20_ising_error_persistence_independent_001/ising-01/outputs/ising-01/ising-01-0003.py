
import subprocess, textwrap
script = r'''
import numpy as np, time, sys
T=float(sys.argv[1]); seed=int(sys.argv[2])
N=16; R=2000; NEQ=3000; NMEAS=20000; BLK=500
rng=np.random.default_rng(seed)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
beta=1/T
ptab=np.zeros(9)
for x in range(-4,5): ptab[x+4]=np.exp(-2*beta*x) if x>0 else 1.0
def sweep(s):
    for m in masks:
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        p=ptab[(s*h+4)]
        flip=(rng.random(s.shape)<p)&m
        s[flip]*=-1
for k in range(NEQ): sweep(s)
nb=NMEAS//BLK
S2=np.zeros((nb,R)); S4=np.zeros((nb,R))
t0=time.time()
for b in range(nb):
    acc2=np.zeros(R); acc4=np.zeros(R)
    for k in range(BLK):
        sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m
        acc2+=m2; acc4+=m2*m2
    S2[b]=acc2; S4[b]=acc4
np.savez(f'/work/T{T}_s{seed}.npz',S2=S2,S4=S4,BLK=BLK,T=T,seed=seed,R=R,NEQ=NEQ,NMEAS=NMEAS)
print('done',T,time.time()-t0,flush=True)
'''
open('/work/run.py','w').write(script)
cmd = "cd /work && (for T in 2.2 1.9 2.0 2.1 2.3 2.4; do python3 run.py $T $((RANDOM)) ; done) > /work/log.txt 2>&1 &"
subprocess.run(cmd, shell=True)
print("launched")
