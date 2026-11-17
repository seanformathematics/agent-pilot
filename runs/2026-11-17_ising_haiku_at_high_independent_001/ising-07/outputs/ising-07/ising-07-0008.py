import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
for T in Ts:
    d=np.load(f'/work/T{T:.2f}.npz')
    n=int(d['meas']); R=len(d['s2'])
    s2=d['s2']/n; s4=d['s4']/n
    tot2=s2.sum(); tot4=s4.sum()
    Ui=1-((tot4-s4)/(R-1))/(3*((tot2-s2)/(R-1))**2)
    Uf=1-(tot4/R)/(3*(tot2/R)**2)
    err=np.sqrt((R-1)/R*np.sum((Ui-Ui.mean())**2))
    print(T, repr(float(Uf)), repr(float(err)), n, R)