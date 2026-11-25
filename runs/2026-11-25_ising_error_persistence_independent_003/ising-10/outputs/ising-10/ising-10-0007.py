import numpy as np
def U_of(a,b): return 1-b.mean()/(3*a.mean()**2)
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/T{T}.npz'); a=d['a']; b=d['b']; R=len(a)
    nb=100; idx=np.arange(R).reshape(nb,-1); Us=[]
    for k in range(nb):
        mask=np.ones(R,bool); mask[idx[k]]=False
        Us.append(U_of(a[mask],b[mask]))
    Us=np.array(Us)
    err=np.sqrt((nb-1)/nb*np.sum((Us-Us.mean())**2))
    print(T, repr(U_of(a,b)), repr(err))