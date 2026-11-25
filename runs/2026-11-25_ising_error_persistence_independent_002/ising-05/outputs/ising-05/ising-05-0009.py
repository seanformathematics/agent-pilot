import numpy as np
files={1.9:'res_T1.900_s191',2.0:'res_T2.000_s201',2.1:'res_T2.100_s211',2.2:'res_T2.200_s221',2.3:'res_T2.300_s231',2.4:'res_T2.400_s241'}
for T,f in files.items():
    d=np.load(f'/work/{f}.npz'); n=int(d['nmeas'])
    a2=d['S2']/n; a4=d['S4']/n; R=len(a2)
    U=lambda idx: 1-a4[idx].mean()/(3*a2[idx].mean()**2)
    full=U(np.arange(R))
    jk=np.array([U(np.delete(np.arange(R),i)) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    print(f"{T} {full:.7f} {err:.7f}")
