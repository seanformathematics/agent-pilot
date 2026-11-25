
script = r'''
import numpy as np, time, sys, json
T=float(sys.argv[1]); seed=int(sys.argv[2]); nmeas=int(sys.argv[3]); nburn=int(sys.argv[4])
L=16; R=1000
rng=np.random.default_rng(seed)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,L,L))
ii,jj=np.meshgrid(np.arange(L),np.arange(L),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s):
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nb
        p=np.exp(-np.maximum(dE,0)/T)
        acc=(rng.random(s.shape,dtype=np.float32)<p)&m
        s[acc]*=-1
    return s
t0=time.time()
for _ in range(nburn): s=sweep(s)
sum2=np.zeros(R); sum4=np.zeros(R); sumabs=np.zeros(R)
blk_m2=[]; blk_m4=[]
B=1000
b2=np.zeros(R); b4=np.zeros(R)
for k in range(nmeas):
    s=sweep(s)
    M=s.sum(axis=(1,2))/(L*L)
    m2=M*M; m4=m2*m2
    sum2+=m2; sum4+=m4; b2+=m2; b4+=m4
    if (k+1)%B==0:
        blk_m2.append(b2/B); blk_m4.append(b4/B); b2=np.zeros(R); b4=np.zeros(R)
m2=sum2/nmeas; m4=sum4/nmeas
np.savez(f'/work/res_T{T}_s{seed}.npz', m2=m2, m4=m4, blk_m2=np.array(blk_m2), blk_m4=np.array(blk_m4))
U_rep = 1-m4/(3*m2**2)
U=1-m4.mean()/(3*m2.mean()**2)
# jackknife over replicas
jk=[]
for r in range(R):
    mask=np.ones(R,bool); mask[r]=False
    jk.append(1-m4[mask].mean()/(3*m2[mask].mean()**2))
jk=np.array(jk)
err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
print(json.dumps(dict(T=T,U=U,U_err=err,time=time.time()-t0, m2=m2.mean(), m4=m4.mean(), U_rep_std=U_rep.std())))
'''
open('/work/ising.py','w').write(script)
import subprocess
r=subprocess.run(['python3','/work/ising.py','1.9','101','45000','5000'],capture_output=True,text=True,timeout=590)
print(r.stdout, r.stderr[-2000:])
