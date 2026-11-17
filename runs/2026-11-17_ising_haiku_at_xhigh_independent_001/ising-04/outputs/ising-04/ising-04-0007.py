import sys; sys.path.insert(0,'/work')
from ising import *
import numpy as np
from scipy.special import ellipk
# exact 4x4 enumeration
N=4
cfg=np.arange(2**(N*N))
bits=((cfg[:,None]>>np.arange(N*N))&1)
S=(2*bits-1).reshape(-1,N,N).astype(np.float64)
E=-(np.sum(S*np.roll(S,-1,1),axis=(1,2))+np.sum(S*np.roll(S,-1,2),axis=(1,2)))
m=S.sum(axis=(1,2))/N**2
def exact_U(T):
    w=np.exp(-(E-E.min())/T); w/=w.sum()
    m2=np.sum(w*m**2); m4=np.sum(w*m**4)
    return 1-m4/(3*m2**2)
masks=make_masks(N)
rng=np.random.default_rng(777)
for T in [1.5,2.0,2.269,2.8]:
    pac=pacc_table(T)
    R=2000
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int32)
    for _ in range(1000): s=sweep(s,pac,masks,rng)
    M2=[];M4=[]
    for _ in range(20000):
        s=sweep(s,pac,masks,rng)
        mm=s.sum(axis=(1,2))/N**2
        M2.append(mm**2); M4.append(mm**4)
    M2=np.array(M2); M4=np.array(M4)
    Us=1-M4.mean()/(3*M2.mean()**2)
    print(f"N=4 T={T}: exact U={exact_U(T):.5f}  MC U={Us:.5f}")
# z-score split-half check for N=16
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def U_idx(bm2,bm4,idx):
    m2=bm2[idx].mean(axis=0); m4=bm4[idx].mean(axis=0)
    return 1-m4.mean()/(3*m2.mean()**2)
def jk(bm2,bm4):
    R=bm2.shape[1]; S2=bm2.mean(axis=0); S4=bm4.mean(axis=0); T2=S2.sum(); T4=S4.sum()
    jkv=np.array([1-((T4-S4[r])/(R-1))/(3*((T2-S2[r])/(R-1))**2) for r in range(R)])
    return np.sqrt((R-1)/R*np.sum((jkv-jkv.mean())**2))
print("N=16 split-half check (interleaved blocks, 4 chunks):")
for T in Ts:
    d=np.load(f'/work/prod_T{T:.2f}.npz'); bm2=d['bm2']; bm4=d['bm4']
    nb=bm2.shape[0]
    full=U_idx(bm2,bm4,slice(None)); err=jk(bm2,bm4)
    # 4 contiguous quarters: scatter of quarter U values vs expected err*2
    qs=[U_idx(bm2,bm4,slice(q*nb//4,(q+1)*nb//4)) for q in range(4)]
    sc=np.std(qs,ddof=1)
    print(f"T={T}: U={full:.5f} jk_err={err:.6f}  quarter-U std={sc:.6f} expected~{err*2:.6f} ratio={sc/(2*err) if err>0 else np.nan:.2f}")
