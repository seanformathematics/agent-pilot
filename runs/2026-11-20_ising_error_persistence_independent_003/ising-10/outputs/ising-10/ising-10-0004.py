
import numpy as np
d=np.load('/work/raw_T1.9_N16.npz')
S2,S4,BLK=d['S2'],d['S4'],int(d['BLK'])
def analyze(S2,S4,burn):
    a2=S2[burn:].sum(0); a4=S4[burn:].sum(0); n=(S2.shape[0]-burn)*BLK
    R=a2.size
    tot2=a2.sum(); tot4=a4.sum(); N_=n*R
    U=1-(tot4/N_)/(3*(tot2/N_)**2)
    Uj=np.array([1-((tot4-a4[r])/((R-1)*n))/(3*((tot2-a2[r])/((R-1)*n))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
for burn in [0,2,5]:
    print(burn,analyze(S2,S4,burn))
