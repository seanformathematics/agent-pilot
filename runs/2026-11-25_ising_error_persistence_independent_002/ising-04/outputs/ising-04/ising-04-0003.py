import numpy as np, time, json, sys
N=16; R=1000; T=float(sys.argv[1]) if len(sys.argv)>1 else 2.3
SEED=12345+int(T*1000)
rng=np.random.default_rng(SEED)
S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def sweep(S,T):
    for m in masks:
        nb=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.int8)
        dE=2*S*nb
        pacc=np.exp(-np.maximum(dE,0)/T).astype(np.float32)
        acc=(dE<=0)|(rng.random(S.shape,dtype=np.float32)<pacc)
        S[acc&m]*=-1
t0=time.time()
BURN=1000; MEAS=int(sys.argv[2]) if len(sys.argv)>2 else 6000
for _ in range(BURN): sweep(S,T)
sm2=np.zeros(R); sm4=np.zeros(R); cnt=0
half2=np.zeros(R); half4=np.zeros(R); hc=0
for k in range(MEAS):
    sweep(S,T)
    m=S.reshape(R,-1).mean(axis=1)
    m2=m*m; m4=m2*m2
    sm2+=m2; sm4+=m4; cnt+=1
    if k==MEAS//2-1: sm2h=sm2.copy(); sm4h=sm4.copy(); cnt_h=cnt
m2r=sm2/cnt; m4r=sm4/cnt
def U_of(a2,a4): return 1-a4.mean()/(3*a2.mean()**2)
U=U_of(m2r,m4r)
# jackknife over replicas
jk=np.empty(R)
idx=np.arange(R)
for r in range(R):
    msk=idx!=r
    jk[r]=U_of(m2r[msk],m4r[msk])
err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
# half-run consistency
Uh=U_of(sm2h/cnt_h,sm4h/cnt_h)
print(json.dumps(dict(T=T,U=U,err=err,Uhalf=Uh,time=time.time()-t0,m2=m2r.mean(),Rmean_m2=float(m2r.mean()))))
np.save(f"/work/rep_T{T}.npy",np.vstack([m2r,m4r]))