
import numpy as np
d=np.load('/work/run_T2.2_N16_s101.npz')
M2,M4=d['M2'],d['M4']
def U_of(m2,m4): return 1-m4.mean()/(3*m2.mean()**2)
def jk(M2,M4,nb=50):
    R=M2.shape[1]
    # per-replica time averages
    a2=M2.mean(0); a4=M4.mean(0)
    U=U_of(a2,a4)
    idx=np.arange(R)%nb
    Us=[]
    for b in range(nb):
        keep=idx!=b
        Us.append(U_of(a2[keep],a4[keep]))
    Us=np.array(Us)
    err=np.sqrt((nb-1)/nb*np.sum((Us-Us.mean())**2))
    return U,err
print("U,err (all):",jk(M2,M4))
# error vs number of time samples used
for n in [100,500,1000,2000]:
    print(n, jk(M2[:n],M4[:n]))
# autocorrelation of replica-averaged m2
x=M2.mean(1); x=x-x.mean()
ac=[np.dot(x[:-l],x[l:])/np.dot(x,x) for l in [1,5,10,20,50,100]]
print("autocorr of <m^2>:",np.round(ac,3))
