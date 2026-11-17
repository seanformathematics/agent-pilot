import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def jk_err_and_U(bm2,bm4):
    R=bm2.shape[1]
    S2=bm2.mean(axis=0); S4=bm4.mean(axis=0)
    T2=S2.sum(); T4=S4.sum()
    U=1-(T4/R)/(3*(T2/R)**2)
    jk=np.array([1-((T4-S4[r])/(R-1))/(3*((T2-S2[r])/(R-1))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return U,err,R
out=[]
for T in Ts:
    d=np.load(f'/work/prod_T{T:.2f}.npz')
    U,err,R=jk_err_and_U(d['bm2'],d['bm4'])
    nsamp=d['bm2'].shape[0]*10*R
    out.append((T,U,err,nsamp,int(d['seed'])))
    print(f"T={T:.1f} U={U:.8f} U_err={err:.3e} n_samples={nsamp}  seed={int(d['seed'])}")
