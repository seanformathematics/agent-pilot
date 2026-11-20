import numpy as np
print([int(T*1000)+11 for T in [1.9,2.0]])
def ustats(S2,S4):
    # S2,S4: (nb,R) block means per replica
    m2=S2.mean(0); m4=S4.mean(0); R=len(m2)
    def U_of(idx):
        a=m2[idx].mean(); b=m4[idx].mean()
        return 1-b/(3*a*a)
    U=U_of(np.arange(R))
    jk=np.array([U_of(np.delete(np.arange(R),r)) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return U,err
for T in [1.9,2.0]:
    d=np.load(f'/work/raw_T{T}.npz')
    S2,S4=d['S2'],d['S4']
    # first-half vs second-half check of equilibration / drift
    U1,_=ustats(S2[:20],S4[:20]); U2,_=ustats(S2[20:],S4[20:])
    U,e=ustats(S2,S4)
    print(T,"U=%.5f err=%.5f"%(U,e),"halves",round(U1,5),round(U2,5))