
import numpy as np
def U(x):
    m2=np.mean(x**2); m4=np.mean(x**4); return 1-m4/(3*m2*m2)
temps=[1.9,2.0,2.1,2.2,2.3,2.4]
for k,T in enumerate(temps):
    a=np.load(f'/work/m_T{T}_s{101+k}.npy')
    res=[]
    for bs in [2000,10000,50000]:
        nb=len(a)//bs
        b=a[:nb*bs].reshape(nb,bs)
        m2=(b**2).mean(1); m4=(b**4).mean(1)
        # jackknife
        S2=m2.sum(); S4=m4.sum()
        Uj=[]
        for i in range(nb):
            M2=(S2-m2[i])/(nb-1); M4=(S4-m4[i])/(nb-1)
            Uj.append(1-M4/(3*M2*M2))
        Uj=np.array(Uj)
        err=np.sqrt((nb-1)/nb*np.sum((Uj-Uj.mean())**2))
        res.append((bs,nb,round(err,6)))
    x=a**2-np.mean(a**2)
    ac=[np.dot(x[:-l],x[l:])/np.dot(x,x) for l in [1,5,10,20]]
    print(T, len(a), "U=%.5f"%U(a), res, "ac",np.round(ac,3))
