import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
for T in Ts:
    o=np.load(f'/work/obs_T{T}_N16.npy')
    B=200; nb=len(o)//B
    blocks=o[:nb*B].reshape(nb,B,3).mean(1)
    tot=blocks.sum(0)
    Uj=np.array([1-((tot-blocks[k])/(nb-1))[1]/(3*((tot-blocks[k])/(nb-1))[0]**2) for k in range(nb)])
    err=np.sqrt((nb-1)/nb*np.sum((Uj-Uj.mean())**2))
    U=1-o[:,1].mean()/(3*o[:,0].mean()**2)
    print(T, repr(float(U)), repr(float(err)), len(o))