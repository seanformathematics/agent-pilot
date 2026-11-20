
import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def U_of(S2,S4,n):
    m2=S2.sum()/n; m4=S4.sum()/n
    return 1-m4/(3*m2*m2)
for T in Ts:
    d=np.load(f'/work/run_T{T}.npz')
    S2,S4=d['S2'],d['S4']; n=int(d['meas'])
    R=len(S2)
    for G in [20,50,100]:
        idx=np.array_split(np.arange(R),G)
        full=U_of(S2,S4,n)
        ests=[]
        for g in idx:
            mask=np.ones(R,bool); mask[g]=False
            ests.append(U_of(S2[mask],S4[mask],n))
        ests=np.array(ests); Gn=len(idx)
        err=np.sqrt((Gn-1)/Gn*np.sum((ests-ests.mean())**2))
        if G==20: e20=err
        print(T,G,round(full,5),'err',round(err,5))
    # naive replica-level error
    u_r=1-S4/(n*3*(S2/n)**2)
    print('   per-replica std/sqrt(R) approx', round(u_r.std(ddof=1)/np.sqrt(R),5))
