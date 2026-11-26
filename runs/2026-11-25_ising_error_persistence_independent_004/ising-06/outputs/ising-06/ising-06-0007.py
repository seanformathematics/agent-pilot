import numpy as np
def ustats(a2,a4):
    n=len(a2)
    U=1-a4.mean()/(3*a2.mean()**2)
    T2=a2.sum(); T4=a4.sum()
    jk=1-((T4-a4)/(n-1))/(3*((T2-a2)/(n-1))**2)
    err=np.sqrt((n-1)/n*np.sum((jk-jk.mean())**2))
    return U,err
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f"/work/res_{T}.npz")
    U,e=ustats(d['a2'],d['a4'])
    print(T, "%.7f %.7f"%(U,e))
