
import numpy as np, glob, json
rows=[]
for T in [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]:
    d=np.load(f'/work/T{T:.2f}.npz')
    s2,s4,MEAS,R=d['s2'],d['s4'],int(d['MEAS']),int(d['R'])
    M2=s2.sum(); M4=s4.sum(); n=R*MEAS
    U=1-(M4/n)/(3*(M2/n)**2)
    # jackknife over replicas
    Ui=np.empty(R)
    for i in range(R):
        a2=(M2-s2[i]); a4=(M4-s4[i])
        Ui[i]=1-(a4/((R-1)*MEAS))/(3*(a2/((R-1)*MEAS))**2)
    err=np.sqrt((R-1)/R*np.sum((Ui-Ui.mean())**2))
    # also a split-half check: 2 groups of 256 replicas
    h1=slice(0,R//2); h2=slice(R//2,R)
    def Uh(sl):
        a2=s2[sl].sum(); a4=s4[sl].sum(); k=len(s2[sl])*MEAS
        return 1-(a4/k)/(3*(a2/k)**2)
    print(f"{T:.2f} U={U:.5f} err_jk={err:.5f} halves={Uh(h1):.4f},{Uh(h2):.4f}")
    rows.append((T,U,err))
