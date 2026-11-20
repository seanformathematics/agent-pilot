import numpy as np
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f"/work/T{T}.npz"); b2=d["m2"]; b4=d["m4"]; n=len(b2)
    U=1-b4.mean()/(3*b2.mean()**2)
    S2=b2.sum(); S4=b4.sum()
    m2=(S2-b2)/(n-1); m4=(S4-b4)/(n-1)
    jk=1-m4/(3*m2**2)
    err=np.sqrt((n-1)/n*np.sum((jk-jk.mean())**2))
    print(T, repr(float(U)), repr(float(err)))
