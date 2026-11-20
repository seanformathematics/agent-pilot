
import numpy as np
def jk_U(a2, a4):
    M2 = a2.mean(); M4 = a4.mean(); U = 1 - M4/(3*M2*M2)
    n = len(a2); t2=a2.sum(); t4=a4.sum()
    q2=(t2-a2)/(n-1); q4=(t4-a4)/(n-1)
    Uj = 1 - q4/(3*q2*q2)
    return U, np.sqrt((n-1)/n*np.sum((Uj-Uj.mean())**2))
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    a2=np.load(f'/work/a2_T{T}.npy'); a4=np.load(f'/work/a4_T{T}.npy')
    U,e=jk_U(a2,a4)
    print(T, "%.6f %.7f"%(U,e))
