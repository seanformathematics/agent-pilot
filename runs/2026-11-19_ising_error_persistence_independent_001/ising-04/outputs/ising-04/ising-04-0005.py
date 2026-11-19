exec(open('/work/ising.py').read())
import time
def U_and_err(S2,S4):
    R=len(S2)
    U = 1 - S4.mean()/(3*S2.mean()**2)
    tot2=S2.sum(); tot4=S4.sum()
    a2=(tot2-S2)/(R-1); a4=(tot4-S4)/(R-1)
    Uj = 1-a4/(3*a2*a2)
    err = np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U, err
for i,T in enumerate([2.2,2.3,2.4]):
    t=time.time()
    s,m2,m4,a = run(T, 4000, 2000, 2000, 2000+i)
    U,e = U_and_err(m2,m4)
    np.savez(f'/work/prod_{T}.npz', S2=m2, S4=m4)
    print(T, round(U,6), round(e,6), round(time.time()-t,1), flush=True)