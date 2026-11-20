
import numpy as np, glob
files=sorted(glob.glob('/work/T*_s*.npz'), key=lambda f: float(f.split('/T')[1].split('_')[0]))
for f in files:
    d=np.load(f)
    S2,S4=d['S2'],d['S4']; BLK=int(d['BLK']); R=int(d['R'])
    nb=S2.shape[0]
    m2=S2.sum(0)/(nb*BLK); m4=S4.sum(0)/(nb*BLK)
    U_of=lambda a2,a4: 1-a4.mean()/(3*a2.mean()**2)
    U=U_of(m2,m4)
    jk=np.array([U_of(np.delete(m2,r),np.delete(m4,r)) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    # block-based error cross-check
    Ub=np.array([1-(S4[b]/BLK).mean()/(3*((S2[b]/BLK).mean())**2) for b in range(nb)])
    print(f"{float(d['T'])} seed={int(d['seed'])} U={U:.7f} err_jk={err:.2e} err_blocks={Ub.std(ddof=1)/np.sqrt(nb):.2e} seed={int(d['seed'])} seed_check")
