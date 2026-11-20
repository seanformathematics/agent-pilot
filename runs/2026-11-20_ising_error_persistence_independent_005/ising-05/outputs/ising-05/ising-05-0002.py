
import numpy as np, time, os, json
os.makedirs('/work',exist_ok=True)
code = r'''
import numpy as np, sys, json, time
N=16
T=float(sys.argv[1]); R=int(sys.argv[2]); neq=int(sys.argv[3]); nmeas=int(sys.argv[4]); seed=int(sys.argv[5])
rng=np.random.default_rng(seed)
beta=1.0/T
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mA=((ii+jj)%2==0); mB=~mA
# random sign start per replica (all-up or all-down, chosen at random) -> symmetric mixture
s=np.where(rng.random((R,1,1))<0.5,1,-1).astype(np.int8)*np.ones((R,N,N),dtype=np.int8)
# add random flips only for high T: start fully random for T>=Tc-ish is also fine; keep ordered start (sign random) -- for T>Tc, ordered start relaxes fast
if T>2.2:
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
# precompute acceptance for dE = 2*s*nn, nn in {-4,-2,0,2,4}; dE in {-8..8}
pacc=np.zeros(17)
for dE in range(-8,9):
    pacc[dE+8]=min(1.0,np.exp(-beta*dE))
def half(s,mask):
    nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
    dE=(2*s*nn).astype(np.int16)
    p=pacc[dE+8]
    acc=(rng.random((R,N,N))<p)&mask
    return np.where(acc,-s,s).astype(np.int8)
def sweep(s):
    s=half(s,mA)
    s=half(s,mB)
    return s
t0=time.time()
for k in range(neq):
    s=sweep(s)
A2=np.zeros(R); A4=np.zeros(R); A1=np.zeros(R)
for k in range(nmeas):
    s=sweep(s)
    m=s.reshape(R,-1).mean(axis=1)
    m2=m*m
    A2+=m2; A4+=m2*m2; A1+=np.abs(m)
n=nmeas
def U_of(A2,A4):
    m2=A2.mean(); m4=A4.mean()
    return 1-m4/(3*m2*m2)
U=U_of(A2/n,A4/n)
# jackknife over replicas
idx=np.arange(R)
tot2=A2.sum(); tot4=A4.sum()
Uj=np.empty(R)
for r in range(R):
    m2=(tot2-A2[r])/((R-1)*n); m4=(tot4-A4[r])/((R-1)*n)
    Uj[r]=1-m4/(3*m2*m2)
Uerr=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
# also block (group-of-replicas) jackknife check: 20 groups
G=20
g=np.array_split(idx,G)
Ug=[]
for gi in g:
    keep=np.ones(R,bool); keep[gi]=False
    m2=A2[keep].sum()/(keep.sum()*n); m4=A4[keep].sum()/(keep.sum()*n)
    Ug.append(1-m4/(3*m2*m2))
Ug=np.array(Ug)
Uerr_g=np.sqrt((G-1)/G*np.sum((Ug-Ug.mean())**2))
res=dict(T=T,R=R,neq=neq,nmeas=nmeas,seed=seed,U=float(U),Uerr=float(Uerr),Uerr_groups=float(Uerr_g),
         absm=float((A1/n).mean()/R), time=time.time()-t0)
print(json.dumps(res))
'''
open('/work/ising.py','w').write(code)
import subprocess
t=time.time()
out=subprocess.run(['python3','/work/ising.py','2.269','1000','200','200','1'],capture_output=True,text=True,timeout=300)
print(out.stdout,out.stderr[-2000:], time.time()-t)
