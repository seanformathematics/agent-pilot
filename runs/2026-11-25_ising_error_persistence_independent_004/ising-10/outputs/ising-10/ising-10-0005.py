
import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def U_of(a2,a4): return 1-a4.mean()/(3*a2.mean()**2)
for T in Ts:
    d=np.load(f'/work/T{T}.npz'); a2=d['a2']; a4=d['a4']; R=len(a2)
    U=U_of(a2,a4)
    idx=np.array_split(np.arange(R),100); n=len(idx)
    Uj=np.array([U_of(np.delete(a2,i),np.delete(a4,i)) for i in idx])
    err=np.sqrt((n-1)/n*np.sum((Uj-Uj.mean())**2))
    print(T, repr(float(U)), repr(float(err)), float(a2.mean()))
