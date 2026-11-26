
import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def U_of(a2,a4): return 1-a4.mean()/(3*a2.mean()**2)
for T in Ts:
    d=np.load(f'/work/T{T}.npz'); a2=d['a2']; a4=d['a4']
    R=len(a2)
    U=U_of(a2,a4)
    for G in [50,100]:
        idx=np.array_split(np.arange(R),G)
        Uj=np.array([U_of(np.delete(a2,i),np.delete(a4,i)) for i in idx])
        n=len(idx)
        err=np.sqrt((n-1)/n*np.sum((Uj-Uj.mean())**2))
        print(T,G,'U=%.5f'%U,'jack err=%.5f'%err)
    # also naive replica-level error check
    # bootstrap check
    rng=np.random.default_rng(7)
    bs=[U_of(a2[k],a4[k]) for k in [rng.integers(0,R,R) for _ in range(400)]]
    print('   bootstrap err=%.5f'%np.std(bs))
