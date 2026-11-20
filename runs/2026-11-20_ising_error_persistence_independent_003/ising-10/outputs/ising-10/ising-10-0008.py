
import numpy as np
N=16
def analyze(S2,S4,BLK,burn):
    a2=S2[burn:].sum(0); a4=S4[burn:].sum(0); nb=S2.shape[0]-burn
    n=nb*BLK; R=a2.size
    tot2=a2.sum(); tot4=a4.sum(); N_=n*R
    U=1-(tot4/N_)/(3*(tot2/N_)**2)
    Uj=np.array([1-((tot4-a4[r])/((R-1)*n))/(3*((tot2-a2[r])/((R-1)*n))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
out={}
for T,fn in [(1.9,'/work/raw_T1.9_N16.npz'),(2.0,'/work/raw_T2.0_N16.npz'),(2.1,'/work/raw_T2.1_N16.npz'),(2.2,'/work/raw_T2.2_N16.npz'),(2.3,'/work/raw_T2.3_N16.npz'),(2.4,'/work/raw_T2.4_N16.npz')]:
    d=np.load(fn); S2,S4,BLK=d['S2'],d['S4'],int(d['BLK'])
    U,e=analyze(S2,S4,BLK,5)
    U0,_=analyze(S2,S4,BLK,0)
    U10,e10=analyze(S2,S4,BLK,10)
    # first vs second half (after burn-in 5)
    h=(S2.shape[0]+5)//2
    Ua,ea=analyze(S2[:h],S4[:h],BLK,5)
    Ub,eb=analyze(S2[h:],S4[h:],BLK,0)
    out[T]=(U,e)
    print(f"T={T}: U={U:.5f} err={e:.2e} | burn0={U0:.5f} burn10={U10:.5f} | 1st half {Ua:.5f}+-{ea:.1e}, 2nd half {Ub:.5f}+-{eb:.1e}")
