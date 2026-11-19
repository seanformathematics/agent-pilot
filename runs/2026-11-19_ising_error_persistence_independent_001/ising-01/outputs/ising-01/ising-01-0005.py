
import numpy as np, glob
def analyze(fn):
    d=np.load(fn)
    S2,S4=d['S2'],d['S4']; blk=int(d['blk'])
    n=S2.shape[0]*blk
    a2=S2.sum(0); a4=S4.sum(0); R=len(a2)
    m2=a2/n; m4=a4/n
    U=1-m4.mean()/(3*m2.mean()**2)
    Ur=1-m4/(3*m2**2)
    tot2=a2.sum(); tot4=a4.sum()
    Uj=np.array([1-((tot4-a4[r])/((R-1)*n))/(3*((tot2-a2[r])/((R-1)*n))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    h=S2.shape[0]//2
    def Ut(sl):
        mm2=S2[sl].sum(0); mm4=S4[sl].sum(0); nn=S2[sl].shape[0]*blk
        return 1-(mm4.mean()/nn)/(3*(mm2.mean()/nn)**2)
    # per-block U estimate scatter (time-block jackknife style) as check
    Ub=[1-(S4[b].mean())/(3*(S2[b].mean())**2) for b in range(S2.shape[0])]
    return dict(file=fn.split('/')[-1],T=float(d['T']),U=round(U,6),err=err,U_h1=round(Ut(slice(0,h)),6),U_h2=round(Ut(slice(h,None)),6),
                Ur_std=Ur.std(),Ur_min=Ur.min(),Ur_max=Ur.max(),m2_mean=m2.mean(), blockU_std=np.std(Ub)/np.sqrt(len(Ub)))
for f in sorted(glob.glob('/work/run_*.npz')):
    print(analyze(f))
