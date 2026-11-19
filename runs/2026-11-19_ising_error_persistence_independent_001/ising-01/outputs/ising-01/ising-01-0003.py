
script = r'''
import numpy as np, time, sys
N=16
i,j=np.indices((N,N))
MASKS=[((i+j)%2==p) for p in (0,1)]
def sweep(s,T,rng):
    for mask in MASKS:
        S = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE = 2*s*S
        p = np.exp(-np.maximum(dE,0)/T)
        u = rng.random(s.shape)
        acc = (dE<=0) | (u<p)
        s[acc & mask] *= -1
T=float(sys.argv[1]); seed=int(sys.argv[2]); R=int(sys.argv[3]); nburn=int(sys.argv[4]); nmeas=int(sys.argv[5]); blk=int(sys.argv[6])
rng=np.random.default_rng(seed)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
for k in range(nburn): sweep(s,T,rng)
nb=nmeas//blk
# per replica, per block: sum m^2, sum m^4, sum |m|, count
S2=np.zeros((nb,R)); S4=np.zeros((nb,R)); S1=np.zeros((nb,R))
t0=time.time()
for b in range(nb):
    a2=np.zeros(R); a4=np.zeros(R); a1=np.zeros(R)
    for k in range(blk):
        sweep(s,T,rng)
        m=s.sum(axis=(1,2),dtype=np.int64)/(N*N)
        m2=m*m
        a2+=m2; a4+=m2*m2; a1+=np.abs(m)
    S2[b]=a2; S4[b]=a4; S1[b]=a1
np.savez(f'/work/run_T{T:.3f}_s{seed}.npz',S2=S2,S4=S4,S1=S1,blk=blk,T=T,seed=seed,R=R,nburn=nburn)
print("done T",T,"time",time.time()-t0)
'''
open('/work/ising.py','w').write(script)
import subprocess
r=subprocess.run(['python3','/work/ising.py','1.9','101','1000','2000','20000','500'],capture_output=True,text=True,timeout=600)
print(r.stdout,r.stderr[-2000:])
