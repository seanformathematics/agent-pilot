
import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
rng=np.random.default_rng(99)
res=np.load('/work/final_results.npy')
for i,T in enumerate(Ts):
    d=np.load(f'/work/prod_{i}.npz')
    M2=d['M2'].astype(np.float64).mean(0); M4=d['M4'].astype(np.float64).mean(0)
    R=M2.size
    U=1-M4.mean()/(3*M2.mean()**2)
    # bootstrap over replicas
    idx=rng.integers(0,R,size=(2000,R))
    b2=M2[idx].mean(1); b4=M4[idx].mean(1)
    Ub=1-b4/(3*b2**2)
    eb=Ub.std(ddof=1)
    # 32-block jackknife over replicas
    nb=32; blocks=np.array_split(np.arange(R),nb)
    jb=[]
    for bl in blocks:
        mask=np.ones(R,bool); mask[bl]=False
        jb.append(1-M4[mask].mean()/(3*M2[mask].mean()**2))
    jb=np.array(jb)
    ej=np.sqrt((nb-1)/nb*np.sum((jb-jb.mean())**2))
    # half-replica split
    Uh1=1-M4[:R//2].mean()/(3*M2[:R//2].mean()**2); Uh2=1-M4[R//2:].mean()/(3*M2[R//2:].mean()**2)
    print(f"T={T:.1f} U={U:.5f} jk={res[i,2]:.5f} boot={eb:.5f} blockjk={ej:.5f} | halfsplit diff={Uh1-Uh2:+.5f}")
