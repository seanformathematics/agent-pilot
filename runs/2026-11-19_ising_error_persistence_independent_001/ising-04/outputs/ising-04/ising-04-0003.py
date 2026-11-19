exec(open('/work/ising.py').read())
import time
def U_and_err(S2,S4):
    R=len(S2)
    U = 1 - S4.mean()/(3*S2.mean()**2)
    # jackknife over replicas
    tot2=S2.sum(); tot4=S4.sum()
    Uj = np.empty(R)
    for k in range(R):
        a2=(tot2-S2[k])/(R-1); a4=(tot4-S4[k])/(R-1)
        Uj[k]=1-a4/(3*a2*a2)
    err = np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U, err
temps=[1.9,2.0,2.1,2.2,2.3,2.4]
for i,T in enumerate(temps):
    t=time.time()
    s,m2,m4,a = run(T, 1000, 1000, 1000, 100+i)
    U,e = U_and_err(m2,m4)
    print(T, round(U,5), round(e,5), round(time.time()-t,1), flush=True)