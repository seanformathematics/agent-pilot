import numpy as np
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/d1_{T}.npz'); m2=d['m2']; m4=d['m4']; nb=100; L=len(m2)//nb
    a2=m2[:L*nb].reshape(nb,L).mean(1); a4=m4[:L*nb].reshape(nb,L).mean(1)
    U=1-m4.mean()/(3*m2.mean()**2)
    Uj=np.array([1-np.delete(a4,i).mean()/(3*np.delete(a2,i).mean()**2) for i in range(nb)])
    err=np.sqrt((nb-1)/nb*np.sum((Uj-Uj.mean())**2))
    print(T, '%.6f'%U, '%.6f'%err, len(m2))
