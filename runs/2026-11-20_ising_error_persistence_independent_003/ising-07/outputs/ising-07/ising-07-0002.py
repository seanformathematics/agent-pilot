
import numpy as np, time, json, sys
N=16
def production(T, R, nburn, nmeas, seed):
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    s2=np.zeros(R); s4=np.zeros(R); cnt=0
    half_m2=[];half_m4=[]
    t0=time.time()
    for sw in range(nburn+nmeas):
        for p in (0,1):
            nn=np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)
            dE=2*S*nn
            pr=np.exp(-np.maximum(dE,0)/T)
            flip=(rng.random((R,N,N))<pr)&masks[p]
            S=np.where(flip,-S,S)
        if sw>=nburn:
            m=S.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2=m*m
            s2+=m2; s4+=m2*m2; cnt+=1
            if cnt<=nmeas//2: half_m2.append(0)  # placeholder
    m2=s2/cnt; m4=s4/cnt
    U_r=1-m4/(3*m2*m2)
    U=U_r.mean()  # average of per-chain U
    Ugl=1-s4.sum()/(3*s2.sum()**2/ R * 0 + 3*(s2.sum()/R*0+1)*0+1) if False else None
    Upool=1-(s4.sum()/R/cnt)/(3*((s2.sum()/R/cnt))**2)
    err=U_r.std(ddof=1)/np.sqrt(R)
    return dict(T=T,R=R,nburn=nburn,nmeas=nmeas,seed=seed,U_mean_chains=float(U),U_pooled=float(Upool),U_err=float(err),time=time.time()-t0)
res=production(2.3,400,3000,27000,12345)
print(res)
json.dump(res,open('/work/res_2.3.json','w'))
