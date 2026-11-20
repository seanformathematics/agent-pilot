
import numpy as np, time, json
N=16
def production(T, R, nburn, nmeas, seed):
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    acc=[np.zeros(R),np.zeros(R)]; acc4=[np.zeros(R),np.zeros(R)]; cnt=[0,0]
    t0=time.time()
    for sw in range(nburn+nmeas):
        for p in (0,1):
            nn=np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)
            dE=2*S*nn
            pr=np.exp(-np.maximum(dE,0)/T)
            flip=(rng.random((R,N,N))<pr)&masks[p]
            S=np.where(flip,-S,S)
        if sw>=nburn:
            h=0 if sw-nburn < nmeas//2 else 1
            m=S.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2=m*m
            acc[h]+=m2; acc4[h]+=m2*m2; cnt[h]+=1
    s2=acc[0]+acc[1]; s4=acc4[0]+acc4[1]; C=cnt[0]+cnt[1]
    m2=s2/C; m4=s4/C
    U_r=1-m4/(3*m2*m2)
    U=U_r.mean()
    err=U_r.std(ddof=1)/np.sqrt(R)
    Uh=[]
    for h in (0,1):
        a2=acc[h]/cnt[h]; a4=acc4[h]/cnt[h]
        Uh.append(float(np.mean(1-a4/(3*a2*a2))))
    return dict(T=T,R=R,nburn=nburn,nmeas=nmeas,seed=seed,U=float(U),U_err=float(err),
                U_firsthalf=Uh[0],U_secondhalf=Uh[1],time=time.time()-t0)
out=[]
for k,T in enumerate([1.9,2.0,2.1,2.2,2.3,2.4]):
    r=production(T,400,3000,15000,1000+k)
    print(r,flush=True)
    out.append(r)
json.dump(out,open('/work/results_all.json','w'),indent=1)
