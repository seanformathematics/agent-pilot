import numpy as np
d=np.load('/work/T2.3.npz'); m2=d['m2']; m4=d['m4']
def U_of(a2,a4): return 1-a4.mean()/(3*a2.mean()**2)
def jk(m2,m4,nb=50):
    idx=np.array_split(np.arange(len(m2)),nb)
    Us=[]
    for b in idx:
        mask=np.ones(len(m2),bool); mask[b]=False
        Us.append(U_of(m2[mask],m4[mask]))
    Us=np.array(Us); n=nb
    return U_of(m2,m4), np.sqrt((n-1)/n*np.sum((Us-Us.mean())**2))
print(jk(m2,m4,50), jk(m2,m4,100))
# also check split-half consistency
print(U_of(m2[:1000],m4[:1000]), U_of(m2[1000:],m4[1000:]))
