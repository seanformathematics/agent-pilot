
import numpy as np, glob
files=sorted(glob.glob('/work/T*_s*.npz'), key=lambda f: float(f.split('/T')[1].split('_')[0]))
for f in files:
    d=np.load(f)
    S2,S4=d['S2'],d['S4']; BLK=int(d['BLK']); NMEAS=int(d['NMEAS']); R=int(d['R'])
    nb=S2.shape[0]
    m2=S2.sum(0)/(nb*BLK); m4=S4.sum(0)/(nb*BLK)   # per replica time averages
    def U_of(a2,a4): return 1-a4.mean()/(3*a2.mean()**2)
    U=U_of(m2,m4)
    # jackknife over replicas
    jk=np.empty(R)
    idx=np.arange(R)
    for r in range(R):
        mask=idx!=r
        jk[r]=U_of(m2[mask],m4[mask])
    Ujk=(R-1)/R*np.sum(jk-jk.mean())**0 if False else U-(R-1)*(jk.mean()-U)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    # block-wise U (time blocks) to check stationarity
    Ub=[]
    for b in range(nb):
        Ub.append(1-(S4[b].mean())/(3*(S2[b].mean())**2)/1.0 if False else 1-(S4[b]/BLK).mean()/(3*((S2[b]/BLK).mean())**2))
    print(f"T={float(d['T'])} U={U:.5f} err(jk)={err:.5f} Ujk_bias_corr={Ujk:.5f} <m^2>={m2.mean():.4f} blocks U first/last: {Ub[0]:.4f} {Ub[-1]:.4f} std_blocks={np.std(Ub):.4f}")
