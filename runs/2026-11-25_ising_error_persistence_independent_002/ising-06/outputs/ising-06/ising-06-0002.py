
import numpy as np, time, json
T=1.9; seed=1001
N=16; R=2000; BURN=1500; MEAS=25000
rng=np.random.default_rng(seed)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s):
    for m in masks:
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nn
        p=np.exp(-np.maximum(dE,0)/T).astype(np.float32)
        acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<p)
        s[acc & m[None]] *= -1
    return s
t0=time.time()
for _ in range(BURN): s=sweep(s)
print("burn done",time.time()-t0,flush=True)
S2=np.zeros(R); S4=np.zeros(R); half=np.zeros((2,R,2))
n=0
M=N*N
for k in range(MEAS):
    s=sweep(s)
    m=s.reshape(R,-1).sum(1)/M
    m2=m*m
    S2+=m2; S4+=m2*m2; n+=1
    if k==MEAS//2-1:
        S2a=S2.copy(); S4a=S4.copy(); na=n
    if time.time()-t0>540: break
print("meas sweeps",n,"time",time.time()-t0,flush=True)
def Ur(S2,S4,n):
    a2=S2/n; a4=S4/n
    return 1-a4/(3*a2*a2)
U_r=Ur(S2,S4,n)
U=U_r.mean()
# jackknife over replicas
jk=np.array([ (np.delete(U_r,i)).mean() for i in range(R)])
err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
# half-split check
Ua=Ur(S2a,S4a,na).mean(); Ub=Ur(S2-S2a,S4-S4a,n-na).mean()
print(json.dumps(dict(T=T,U=U,err=err,n=n,R=R,Ufirsthalf=Ua,Usecondhalf=Ub)))
json.dump(dict(T=T,seed=seed,n=n,R=R,S2=S2.tolist(),S4=S4.tolist()),open(f'/work/res_T{T}.json','w'))
