
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
code = r'''
import numpy as np, sys, time
N=16
T=float(sys.argv[1]); seed=int(sys.argv[2]); R=int(sys.argv[3]); nEq=int(sys.argv[4]); nMeas=int(sys.argv[5]); out=sys.argv[6]
rng=np.random.default_rng(seed)
s=np.where(rng.random((R,N,N))<0.5,-1,1).astype(np.int8)
# acceptance table indexed by h = sum of neighbours (-4..4) and s
# dE = 2*s*h ; accept prob min(1,exp(-dE/T))
pacc={}
mask0=((np.arange(N)[:,None]+np.arange(N)[None,:])%2==0)
masks=[mask0,~mask0]
def sweep(s):
    for mk in masks:
        h=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=2*s*h
        p=np.exp(-np.clip(dE,0,None)/T)
        u=rng.random(s.shape)
        flip=(u<p)&mk
        s=np.where(flip,-s,s).astype(np.int8)
    return s
t0=time.time()
for _ in range(nEq): s=sweep(s)
S2=np.zeros(R); S4=np.zeros(R); S1=np.zeros(R); E=np.zeros(R)
m_series=[]
for k in range(nMeas):
    s=sweep(s)
    m=s.reshape(R,-1).mean(1)
    am=np.abs(m)
    S2+=m*m; S4+=m**4; S1+=am
    if k%10==0:
        pass
np.savez(out,S2=S2/nMeas,S4=S4/nMeas,S1=S1/nMeas,T=T,seed=seed,R=R,nEq=nEq,nMeas=nMeas)
print('done',T,time.time()-t0)
'''
open('/work/ising.py','w').write(code)
import subprocess
t=time.time()
r=subprocess.run(['python3','/work/ising.py','2.269','1','400','100','100','/tmp/test.npz'],capture_output=True,text=True)
print(r.stdout,r.stderr,time.time()-t)
