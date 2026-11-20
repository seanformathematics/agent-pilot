
import numpy as np
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/T{T}.npz'); S2=d['S2']; S4=d['S4']; n=float(d['n'])
    R=len(S2)
    A=S2.sum(); B=S4.sum()
    U=1-(B/n/R)/(3*(A/n/R)**2)
    # leave-one-replica-out
    Ai=A-S2; Bi=B-S4
    Ui=1-(Bi/n/(R-1))/(3*(Ai/n/(R-1))**2)
    err=np.sqrt((R-1)/R*np.sum((Ui-Ui.mean())**2))
    # also check with 20 groups
    G=20; idx=np.arange(R).reshape(G,-1)
    Ug=[]
    for g in range(G):
        m=np.ones(R,bool); m[idx[g]]=False
        Ug.append(1-(B-S4[idx[g]].sum())/((n*(R-R//G)))/(3*((A-S2[idx[g]].sum())/(n*(R-R//G)))**2))
    Ug=np.array(Ug); errg=np.sqrt((G-1)/G*np.sum((Ug-Ug.mean())**2))
    print(f"T={T} U={U:.5f} err_jk_rep={err:.5f} err_jk20={errg:.5f}")
