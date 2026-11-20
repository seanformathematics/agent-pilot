import numpy as np, glob
def U_of(m2,m4): return 1-np.mean(m4)/(3*np.mean(m2)**2)
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    f=glob.glob(f"/work/M_{T}_*.npy")[0]
    M=np.load(f); x=(M/256.0)**2; x2=x*x
    res=[]
    for B in [1000,5000,20000,100000]:
        nb=len(x)//B
        a=x[:nb*B].reshape(nb,B).mean(1); b=x2[:nb*B].reshape(nb,B).mean(1)
        Sa=a.sum(); Sb=b.sum()
        Ul=[]
        for i in range(nb):
            ma=(Sa-a[i])/(nb-1); mb=(Sb-b[i])/(nb-1)
            Ul.append(1-mb/(3*ma*ma))
        Ul=np.array(Ul)
        err=np.sqrt((nb-1)/nb*np.sum((Ul-Ul.mean())**2))
        res.append((B,nb,round(err,5)))
    print(T, round(U_of(x,x2),5), res)
