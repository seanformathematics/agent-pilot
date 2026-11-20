import numpy as np
def U_of(a2,a4): return 1-a4.mean()/(3*a2.mean()**2)
def jk(m2,m4,nb=50):
    idx=np.array_split(np.arange(len(m2)),nb)
    Us=[]
    for b in idx:
        mask=np.ones(len(m2),bool); mask[b]=False
        Us.append(U_of(m2[mask],m4[mask]))
    Us=np.array(Us); n=nb
    return U_of(m2,m4), np.sqrt((n-1)/n*np.sum((Us-Us.mean())**2))
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/T{T}.npz')
    U,e=jk(d['m2'],d['m4'])
    print(T, round(U,6), e)
