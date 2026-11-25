
script = r'''
import numpy as np, time, os, sys
N=16; R=2000; BLK=100
T=float(sys.argv[1]); tlimit=float(sys.argv[2]); total_meas=int(sys.argv[3]); burn=int(sys.argv[4])
seed=int(sys.argv[5])
fn=f'/work/state_T{T:.3f}.npz'
i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
EV=((i+j)%2==0); OD=~EV
if os.path.exists(fn):
    d=np.load(fn); s=d['s']; sweeps=int(d['sweeps']); A=d['A']; B=d['B']; C=d['C']; rng=np.random.default_rng([seed,sweeps])
else:
    rng=np.random.default_rng([seed,0])
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8); sweeps=0
    A=np.zeros((R,0)); B=np.zeros((R,0)); C=np.zeros((R,0))
p_table=np.exp(-4.0*np.arange(0,9)/T)  # index dE/2? we use dE directly
def sweep(s):
    for mask in (EV,OD):
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nn
        pr=np.exp(-np.maximum(dE,0)/T)
        acc=(dE<=0)|(rng.random(s.shape)<pr)
        s[acc&mask]*=-1
    return s
t0=time.time()
bA=[];bB=[];bC=[]
acc_s=np.zeros(R); acc2=np.zeros(R); acc4=np.zeros(R); accabs=np.zeros(R); cnt=0
target = burn + total_meas
while sweeps < target and time.time()-t0 < tlimit:
    s=sweep(s); sweeps+=1
    if sweeps>burn:
        m=s.reshape(R,-1).mean(1)
        m2=m*m
        acc2+=m2; acc4+=m2*m2; accabs+=np.abs(m); cnt+=1
        if cnt==BLK:
            bA.append(acc2/BLK); bB.append(acc4/BLK); bC.append(accabs/BLK)
            acc2=np.zeros(R); acc4=np.zeros(R); accabs=np.zeros(R); cnt=0
    if sweeps%1000==0 and sweeps<=burn:
        pass
# save
A2=np.concatenate([A,np.array(bA).T.reshape(R,-1)],axis=1) if bA else A
B2=np.concatenate([B,np.array(bB).T.reshape(R,-1)],axis=1) if bB else B
C2=np.concatenate([C,np.array(bC).T.reshape(R,-1)],axis=1) if bC else C
# keep partial block accumulators too (only if burn done)
np.savez(fn,s=s,sweeps=sweeps,A=A2,B=B2,C=C2)
print('T',T,'sweeps',sweeps,'target',target,'blocks',A2.shape[1],'elapsed',round(time.time()-t0,1))
'''
open('/work/run_ising.py','w').write(script)
import subprocess
r=subprocess.run(['python3','/work/run_ising.py','2.269','500','20000','3000','101'],capture_output=True,text=True,timeout=600)
print(r.stdout,r.stderr[-2000:])
