
import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
for T in Ts:
    d=np.load(f'/work/res_{T}.npz'); m2=d['m2']; m4=d['m4']; R=len(m2)
    U=1-m4.mean()/(3*m2.mean()**2)
    # jackknife over replicas (delete-one via sums)
    S2=m2.sum(); S4=m4.sum()
    loo=np.array([1-((S4-m4[i])/(R-1))/(3*((S2-m2[i])/(R-1))**2) for i in range(R)])
    se=np.sqrt((R-1)/R*np.sum((loo-loo.mean())**2))
    print(f"T={T} U={U:.5f} err={se:.5f}")
