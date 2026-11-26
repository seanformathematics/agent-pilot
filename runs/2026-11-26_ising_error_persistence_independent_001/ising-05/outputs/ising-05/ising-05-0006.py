import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def U_of(o):
    m2=o[:,0].mean(); m4=o[:,1].mean()
    return 1-m4/(3*m2**2)
for T in Ts:
    o=np.load(f'/work/obs_T{T}_N16.npy')
    U=U_of(o)
    res={}
    for B in [100,200,500,1000]:
        nb=len(o)//B
        blocks=o[:nb*B].reshape(nb,B,3).mean(1)
        # jackknife
        tot=blocks.sum(0)
        Uj=[]
        for k in range(nb):
            mm=(tot-blocks[k])/(nb-1)
            Uj.append(1-mm[1]/(3*mm[0]**2))
        Uj=np.array(Uj)
        err=np.sqrt((nb-1)/nb*np.sum((Uj-Uj.mean())**2))
        res[B]=round(err,5)
    h1=U_of(o[:10000]); h2=U_of(o[10000:])
    print(T, round(U,5), res, 'halves', round(h1,4), round(h2,4))