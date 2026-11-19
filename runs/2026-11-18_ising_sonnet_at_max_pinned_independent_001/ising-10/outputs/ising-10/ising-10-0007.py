
import sys, time
import numpy as np
sys.path.insert(0,'/work')
from sw import SWReplicas
from ana import jackknife_U

N=4; nsite=16
states = ((np.arange(2**nsite)[:,None] >> np.arange(nsite)[None,:]) & 1)*2-1
grid = states.reshape(-1,N,N).astype(np.int32)
E = -(grid*(np.roll(grid,-1,axis=2)+np.roll(grid,-1,axis=1))).sum(axis=(1,2))
m = grid.sum(axis=(1,2))/nsite
def exact(T):
    w=np.exp(-(E-E.min())/T); Z=w.sum()
    m2=(w*m**2).sum()/Z; m4=(w*m**4).sum()/Z
    return 1-m4/(3*m2**2), (w*E).sum()/Z/nsite

R=500; nburn=50; nmeas=1000; nb=12
zs_all=[]; zes_all=[]
t0=time.time()
for T in [2.269,2.5,3.0]:
    U_ex,e_ex=exact(T)
    zs=[]; zes=[]; Us=[]; errs=[]
    for b in range(nb):
        rng=np.random.default_rng(31000+int(T*1000)+b)
        sim=SWReplicas(N,R,T,rng,'hot')
        for _ in range(nburn): sim.sweep()
        S2=np.zeros(R); S4=np.zeros(R); SE=np.zeros(R)
        for _ in range(nmeas):
            sim.sweep()
            mm=sim.magnetization_sum()/nsite
            S2+=mm**2; S4+=mm**4; SE+=sim.energy()
        U,err,_=jackknife_U(S2,S4,nmeas)
        e_r=SE/nmeas/nsite
        zs.append((U-U_ex)/err); zes.append((e_r.mean()-e_ex)/(e_r.std(ddof=1)/np.sqrt(R)))
        Us.append(U); errs.append(err)
    zs=np.array(zs); zes=np.array(zes); Us=np.array(Us)
    print(f"T={T}: U_exact={U_ex:.5f}; U scatter over batches={Us.std(ddof=1):.5f} vs mean jackknife err={np.mean(errs):.5f}; "
          f"z(U): mean={zs.mean():+.2f} std={zs.std(ddof=1):.2f} ; z(e): mean={zes.mean():+.2f} std={zes.std(ddof=1):.2f}  [{time.time()-t0:.0f}s]")
    zs_all+=list(zs); zes_all+=list(zes)
zs_all=np.array(zs_all); zes_all=np.array(zes_all)
print("ALL: z(U) mean %.2f (+-%.2f) std %.2f ; z(e) mean %.2f (+-%.2f) std %.2f ; n=%d"%(zs_all.mean(),1/np.sqrt(len(zs_all)),zs_all.std(ddof=1),zes_all.mean(),1/np.sqrt(len(zes_all)),zes_all.std(ddof=1),len(zs_all)))
