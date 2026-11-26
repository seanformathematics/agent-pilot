import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]; K=2000
d=np.load('/work/meas_1.npz'); M2=d['M2']; M4=d['M4']
def U_of(m2,m4): return 1-m4.mean()/(3*m2.mean()**2)
def jack(a,b,G=100):
    idx=np.arange(a.shape[1])%G
    full=U_of(a,b)
    loo=np.array([U_of(a[:,idx!=g],b[:,idx!=g]) for g in range(G)])
    return full, np.sqrt((G-1)/G*np.sum((loo-loo.mean())**2))
for t,T in enumerate(Ts):
    a=M2[:,t*K:(t+1)*K]; b=M4[:,t*K:(t+1)*K]
    f,e=jack(a,b)
    print(T, repr(float(f)), repr(float(e)), K, M2.shape[0]*100)