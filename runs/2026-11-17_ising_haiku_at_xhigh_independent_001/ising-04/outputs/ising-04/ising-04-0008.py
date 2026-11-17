import sys; sys.path.insert(0,'/work')
from ising import *
import numpy as np
N=4
cfg=np.arange(2**(N*N))
bits=((cfg[:,None]>>np.arange(N*N))&1)
S=(2*bits-1).reshape(-1,N,N).astype(np.float64)
E=-(np.sum(S*np.roll(S,-1,1),axis=(1,2))+np.sum(S*np.roll(S,-1,2),axis=(1,2)))
m=S.sum(axis=(1,2))/N**2
masks=make_masks(N)
for T in [2.0,2.8]:
    w=np.exp(-(E-E.min())/T); w/=w.sum()
    ex=dict(E=np.sum(w*E), m2=np.sum(w*m**2), m4=np.sum(w*m**4), am=np.sum(w*np.abs(m)))
    ex['U']=1-ex['m4']/(3*ex['m2']**2)
    rng=np.random.default_rng(99)
    pac=pacc_table(T)
    R=2000
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int32)
    for _ in range(1000): s=sweep(s,pac,masks,rng)
    Es=[];M2=[];M4=[];AM=[]
    for _ in range(10000):
        s=sweep(s,pac,masks,rng)
        mm=s.sum(axis=(1,2))/N**2
        e=-(np.sum(s*np.roll(s,-1,1),axis=(1,2))+np.sum(s*np.roll(s,-1,2),axis=(1,2)))
        Es.append(e.astype(float)); M2.append(mm**2); M4.append(mm**4); AM.append(np.abs(mm))
    Es=np.array(Es);M2=np.array(M2);M4=np.array(M4);AM=np.array(AM)
    # per-replica means for error bars
    pe=Es.mean(axis=0); p2=M2.mean(axis=0); p4=M4.mean(axis=0); pa=AM.mean(axis=0)
    se=lambda x: x.std(ddof=1)/np.sqrt(len(x))
    print(f"T={T}: E mc={pe.mean():.4f}±{se(pe):.4f} exact={ex['E']:.4f}")
    print(f"      m2 mc={p2.mean():.5f}±{se(p2):.5f} exact={ex['m2']:.5f}")
    print(f"      m4 mc={p4.mean():.5f}±{se(p4):.5f} exact={ex['m4']:.5f}")
    print(f"      |m| mc={pa.mean():.5f}±{se(pa):.5f} exact={ex['am']:.5f}")
    Umc=1-p4.mean()/(3*p2.mean()**2)
    print(f"      U mc={Umc:.5f} exact={ex['U']:.5f}")
