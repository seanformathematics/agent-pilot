
import numpy as np
d=np.load('/work/meas.npz'); M2=d['m2']; M4=d['m4']
temps=[1.9,2.0,2.1,2.2,2.3,2.4]; R_per=1000
def U_of(m2r,m4r):
    a=m2r.mean(); b=m4r.mean(); return 1-b/(3*a*a)
for k,T in enumerate(temps):
    sl=slice(k*R_per,(k+1)*R_per)
    m2r=M2[:,sl].mean(0); m4r=M4[:,sl].mean(0)
    U=U_of(m2r,m4r)
    G=50; groups=np.array_split(np.arange(R_per),G); jk=[]
    for g in groups:
        keep=np.ones(R_per,bool); keep[g]=False
        jk.append(U_of(m2r[keep],m4r[keep]))
    jk=np.array(jk); err=np.sqrt((G-1)/G*np.sum((jk-jk.mean())**2))
    print(T, repr(float(U)), repr(float(err)))
