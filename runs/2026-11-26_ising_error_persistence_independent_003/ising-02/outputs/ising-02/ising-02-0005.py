
import numpy as np
d=np.load('/work/meas.npz'); M2=d['m2']; M4=d['m4']   # (nblocks, R)
temps=[1.9,2.0,2.1,2.2,2.3,2.4]; R_per=1000
print("blocks",M2.shape)
def U_of(m2r,m4r):
    a=m2r.mean(); b=m4r.mean()
    return 1-b/(3*a*a)
for k,T in enumerate(temps):
    sl=slice(k*R_per,(k+1)*R_per)
    m2=M2[:,sl]; m4=M4[:,sl]
    m2r=m2.mean(0); m4r=m4.mean(0)
    U=U_of(m2r,m4r)
    # replica jackknife with G groups
    G=50
    idx=np.arange(R_per); groups=np.array_split(idx,G)
    jk=[]
    for g in groups:
        keep=np.ones(R_per,bool); keep[g]=False
        jk.append(U_of(m2r[keep],m4r[keep]))
    jk=np.array(jk)
    err=np.sqrt((G-1)/G*np.sum((jk-jk.mean())**2))
    # drift: first half vs second half of measurement
    h=M2.shape[0]//2
    U1=U_of(m2[:h].mean(0),m4[:h].mean(0)); U2=U_of(m2[h:].mean(0),m4[h:].mean(0))
    print(f"T={T} U={U:.5f} err={err:.5f}  halves: {U1:.5f} {U2:.5f}")
