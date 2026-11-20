
import numpy as np
d=np.load('/work/state.npz')
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
S2=d['S2']; S4=d['S4']; cnt=int(d['cnt'])
print("cnt",cnt)
for k,T in enumerate(Ts):
    m2=S2[k]/cnt; m4=S4[k]/cnt
    U=1-np.mean(m4)/(3*np.mean(m2)**2)
    R=len(m2)
    # jackknife over chains
    tot2=S2[k].sum(); tot4=S4[k].sum()
    Uj=np.empty(R)
    for c in range(R):
        a2=(tot2-S2[k,c])/((R-1)*cnt); a4=(tot4-S4[k,c])/((R-1)*cnt)
        Uj[c]=1-a4/(3*a2**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    print(T, round(U,5), round(err,5))
