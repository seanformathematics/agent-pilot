
import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
rng=np.random.default_rng(99)
final=[]
for i,T in enumerate(Ts):
    d=np.load(f'/work/prod_{i}.npz')
    M2=d['M2'].astype(np.float64).mean(0); M4=d['M4'].astype(np.float64).mean(0)
    R=M2.size
    U=1-M4.mean()/(3*M2.mean()**2)
    m2r=M2; m4r=M4
    tot2=m2r.sum(); tot4=m4r.sum()
    a2=(tot2-m2r)/(R-1); a4=(tot4-m4r)/(R-1)
    jk=1-a4/(3*a2**2)
    ejk=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    idx=rng.integers(0,R,size=(2000,R))
    Ub=1-M4[idx].mean(1)/(3*M2[idx].mean(1)**2); eb=Ub.std(ddof=1)
    nb=32; blocks=np.array_split(np.arange(R),nb); jb=[]
    for bl in blocks:
        mask=np.ones(R,bool); mask[bl]=False
        jb.append(1-M4[mask].mean()/(3*M2[mask].mean()**2))
    jb=np.array(jb); ebj=np.sqrt((nb-1)/nb*np.sum((jb-jb.mean())**2))
    e=max(ejk,eb,ebj)
    final.append((T,U,e))
    print(f"T={T} U={U:.7f} jk={ejk:.3e} boot={eb:.3e} blockjk={ebj:.3e} reported={e:.3e} nsamp={R*2000}")
