
import numpy as np
def jk(S2,S4,n):
    R=len(S2)
    tot2=S2.sum(); tot4=S4.sum()
    U=1-(tot4/(R*n))/(3*(tot2/(R*n))**2)
    # leave-one-out
    a2=(tot2-S2)/((R-1)*n); a4=(tot4-S4)/((R-1)*n)
    Ui=1-a4/(3*a2**2)
    err=np.sqrt((R-1)/R*np.sum((Ui-Ui.mean())**2))
    return U,err
for T in [1.9,2.0]:
    d=np.load(f"/work/sums_T{T}.npz")
    print(T,jk(d['S2'],d['S4'],int(d['n'])))
