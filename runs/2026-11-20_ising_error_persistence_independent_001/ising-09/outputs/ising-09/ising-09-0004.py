
import numpy as np, os
print(os.listdir('/work'))
def U_of(M2,M4): return 1-M4.mean()/(3*M2.mean()**2)
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    M=np.load(f"/work/pilot_{T}.npy").astype(np.float64)
    nm,R=M.shape
    M2=(M**2).mean(axis=0); M4=(M**4).mean(axis=0)   # per-replica time averages
    # jackknife over replicas
    tot2=M2.sum(); tot4=M4.sum()
    Uj=np.array([1-((tot4-M4[r])/(R-1))/(3*((tot2-M2[r])/(R-1))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # autocorrelation of m^2 averaged over replicas
    x=(M**2).mean(axis=1); x=x-x.mean()
    ac=[np.dot(x[:-k],x[k:])/np.dot(x,x) if k>0 else 1 for k in range(0,400,5)]
    ac=np.array(ac); 
    tau=1+2*np.sum(ac[1:][ac[1:]>0])*5
    print(T,"U=%.4f err_jk=%.5f"%(U_of(M2,M4),err),"tau_m2~",round(tau,1))
