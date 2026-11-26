
import numpy as np, pickle
d=pickle.load(open('/work/all.pkl','rb'))
b2=d['b2']; b4=d['b4']
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]; R=1000; nb=b2.shape[0]
def U_of(reps):
    s2=b2[:,reps].sum(); s4=b4[:,reps].sum()
    n=nb*len(reps)*100
    m2=s2/n; m4=s4/n
    return 1-m4/(3*m2**2)
for k,T in enumerate(Ts):
    reps=np.arange(k*R,(k+1)*R)
    U=U_of(reps)
    G=50; groups=np.array_split(reps,G)
    jk=np.array([U_of(np.setdiff1d(reps,g)) for g in groups])
    err=np.sqrt((G-1)/G*np.sum((jk-jk.mean())**2))
    print(f"{T} {U:.6f} {err:.7f} {nb*100*R}")
