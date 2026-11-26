
import numpy as np
d=np.load('/work/meas.npz'); M2=d['m2']; M4=d['m4']
temps=[1.9,2.0,2.1,2.2,2.3,2.4]; R_per=1000
def U_of(m2r,m4r):
    a=m2r.mean(); b=m4r.mean(); return 1-b/(3*a*a)
for k,T in enumerate(temps):
    sl=slice(k*R_per,(k+1)*R_per)
    m2r=M2[:,sl].mean(0); m4r=M4[:,sl].mean(0)
    Ug=np.array([U_of(m2r[g*100:(g+1)*100],m4r[g*100:(g+1)*100]) for g in range(10)])
    print(T, "U mean of groups",Ug.mean().round(5),"sd_group",Ug.std(ddof=1).round(5),"err of full mean ~",(Ug.std(ddof=1)/np.sqrt(10)).round(5))
