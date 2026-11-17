import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def U_of(x):
    m2=np.mean(x**2); m4=np.mean(x**4); return 1-m4/(3*m2*m2)
for T in Ts:
    x=np.load(f'/work/pilot_{T}.npy')
    out=[]
    for bs in [1000,10000,50000]:
        nb=len(x)//bs
        bins=x[:nb*bs].reshape(nb,bs)
        Ub=np.array([U_of(np.delete(bins,b,axis=0).ravel()) for b in range(nb)])
        out.append(np.sqrt((nb-1)/nb*np.sum((Ub-Ub.mean())**2)))
    print(T, round(U_of(x),5), [round(v,5) for v in out])
