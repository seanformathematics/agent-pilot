import numpy as np, glob
from scipy.special import ellipk
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def onsager_u(T):
    K=1/T
    k=2*np.sinh(2*K)/np.cosh(2*K)**2
    return -(1/np.tanh(2*K))*(1+(2/np.pi)*(2*np.tanh(2*K)**2-1)*ellipk(k**2))
def U_from(bm2,bm4, idx=None):
    if idx is None: idx=slice(None)
    m2=bm2[idx].mean(axis=0); m4=bm4[idx].mean(axis=0)  # per replica
    return 1-m4.mean()/(3*m2.mean()**2)
def jk_err(bm2,bm4):
    R=bm2.shape[1]
    S2=bm2.mean(axis=0); S4=bm4.mean(axis=0)  # per-replica means
    T2=S2.sum(); T4=S4.sum()
    jk=np.array([1-((T4-S4[r])/(R-1))/(3*((T2-S2[r])/(R-1))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return err
rows=[]
for T in Ts:
    d=np.load(f'/work/prod_T{T:.2f}.npz')
    bm2=d['bm2']; bm4=d['bm4']; be=d['be']
    U=U_from(bm2,bm4); err=jk_err(bm2,bm4)
    # halves
    nb=bm2.shape[0]
    U1=U_from(bm2,bm4,slice(0,nb//2)); U2=U_from(bm2,bm4,slice(nb//2,None))
    # first 10% vs rest (burn-in check)
    U_early=U_from(bm2,bm4,slice(0,30)); 
    rows.append((T,U,err,U1,U2,U_early,be.mean(),onsager_u(T)))
    print(f"T={T:.2f} U={U:.5f} err={err:.6f} | halves {U1:.5f} {U2:.5f} | first300sweeps {U_early:.5f} | e={be.mean():.5f} onsager_e={-(1/np.tanh(2/T))*(1+(2/np.pi)*(2*np.tanh(2/T)**2-1)*ellipk((2*np.sinh(2/T)/np.cosh(2/T)**2)**2)):.5f}")
