
import numpy as np
M=np.load('/work/M16.npy').astype(np.float64)
Ts=np.linspace(1.5,2.8,14); per=100
def U_of(m2,m4):
    return 1-np.mean(m4)/(3*np.mean(m2)**2)
out=[]
for i,T in enumerate(Ts):
    X=M[:,i*per:(i+1)*per]
    m2=(X**2).mean(0); m4=(X**4).mean(0)   # per replica
    U=U_of(m2,m4)
    # replica jackknife
    jk=np.empty(per)
    for r in range(per):
        k=np.delete(np.arange(per),r)
        jk[r]=U_of(m2[k],m4[k])
    err=np.sqrt((per-1)/per*np.sum((jk-jk.mean())**2))
    # halves check
    h=X.shape[0]//2
    Uh1=U_of((X[:h]**2).mean(0),(X[:h]**4).mean(0)); Uh2=U_of((X[h:]**2).mean(0),(X[h:]**4).mean(0))
    out.append((T,U,err))
    print(f"T={T:.2f} U={U:.5f} err={err:.5f}  halves {Uh1:.4f} {Uh2:.4f}  <|m|>={np.mean(np.abs(X)):.4f}")
