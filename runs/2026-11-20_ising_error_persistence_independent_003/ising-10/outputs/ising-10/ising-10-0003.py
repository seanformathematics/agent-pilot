
import numpy as np
d=np.load('/work/raw_T1.9_N16.npz')
S2,S4,BLK=d['S2'],d['S4'],int(d['BLK'])
def analyze(S2,S4,burn):
    a2=S2[burn:].sum(0); a4=S4[burn:].sum(0); n=(S2.shape[0]-burn)*BLK
    R=a2.size
    U=1-(a4.sum()/n)/(3*(a2.sum()/n)**2)
    # leave-one-chain-out jackknife
    Uj=np.empty(R)
    tot2=a2.sum(); tot4=a4.sum()
    for r in range(R):
        x2=(tot2-a2[r])/((R-1)*n); x4=(tot4-a4[r])/((R-1)*n)
        Uj[r]=1-x4/(3*x2*x2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
for burn in [0,2,5]:
    print(burn,analyze(S2,S4,burn))
# error estimate with chain splitting into groups of blocks (sanity)
