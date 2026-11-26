import numpy as np
def U_of(S2,S4):
    m2=S2.mean(axis=-1); m4=S4.mean(axis=-1)
    return 1-m4/(3*m2**2)
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/T{T:.2f}.npz'); S2=d['S2']; S4=d['S4']; C=S2.shape[0]
    U=U_of(S2.mean(0)[None],S4.mean(0)[None])[0]
    jk=np.array([U_of(np.delete(S2,c,0).mean(0)[None],np.delete(S4,c,0).mean(0)[None])[0] for c in range(C)])
    err=np.sqrt((C-1)/C*np.sum((jk-jk.mean())**2))
    print(T, repr(U), repr(err), S2.shape)
