import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def binder(a2,a4): return 1-a4/(3*a2**2)
for T in Ts:
    d=np.load(f"/work/res/T{T:.2f}.npz")
    a2=d["a2"]; a4=d["a4"]; R=len(a2)
    U=binder(a2.mean(),a4.mean())
    tot2=a2.sum(); tot4=a4.sum()
    Uj=binder((tot2-a2)/(R-1),(tot4-a4)/(R-1))
    ej=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    print(f"{T:.1f} U={U:.8f} err={ej:.3e} seed={int(d['seed'])}")
