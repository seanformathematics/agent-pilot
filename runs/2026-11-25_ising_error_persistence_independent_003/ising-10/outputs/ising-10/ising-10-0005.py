import numpy as np
def U_of(a,b): return 1-b.mean()/(3*a.mean()**2)
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/T{T}.npz'); a=d['a']; b=d['b']; R=len(a)
    U=U_of(a,b)
    # delete-block jackknife over replicas (100 blocks of 20)
    nb=100; idx=np.arange(R).reshape(nb,-1)
    Us=[]
    for k in range(nb):
        mask=np.ones(R,bool); mask[idx[k]]=False
        Us.append(U_of(a[mask],b[mask]))
    Us=np.array(Us)
    err=np.sqrt((nb-1)/nb*np.sum((Us-Us.mean())**2))
    # bootstrap check
    rng=np.random.default_rng(7)
    bs=[U_of(a[i],b[i]) for i in (rng.integers(0,R,(400,R)))]
    print(T, round(U,5), "jk err", round(err,5), "boot err", round(np.std(bs),5), "R",R)