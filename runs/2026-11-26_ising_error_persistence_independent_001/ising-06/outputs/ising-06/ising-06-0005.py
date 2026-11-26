import numpy as np
Ts=np.array([1.9,2.0,2.1,2.2,2.3,2.4]); K=2000
d=np.load('/work/meas_1.npz'); M2=d['M2']; M4=d['M4']
def U_of(m2,m4): return 1-m4.mean()/(3*m2.mean()**2)
def jack(M2t,M4t,G=100):
    # M2t: (nblk, K) chains
    nb,k=M2t.shape
    idx=np.arange(k)%G
    full=U_of(M2t,M4t)
    loo=[]
    for g in range(G):
        sel=idx!=g
        loo.append(U_of(M2t[:,sel],M4t[:,sel]))
    loo=np.array(loo)
    var=(G-1)/G*np.sum((loo-loo.mean())**2)
    return full, np.sqrt(var)
for t,T in enumerate(Ts):
    a=M2[:,t*K:(t+1)*K]; b=M4[:,t*K:(t+1)*K]
    f,e=jack(a,b)
    # halves
    f1,_=jack(a[:20],b[:20]); f2,_=jack(a[20:],b[20:])
    print(f"T={T} U={f:.5f} err={e:.5f}  first-half-blocks {f1:.4f} second {f2:.4f}")