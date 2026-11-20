
import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def U_of(M):  # M: (steps, R)
    m2=(M**2).mean(axis=0); m4=(M**4).mean(axis=0)
    return 1-m4.mean()/(3*m2.mean()**2)
for T in Ts:
    M=np.load(f'/work/pilot_T{T}.npy').astype(np.float64)
    R=M.shape[1]
    # per-replica sums for jackknife
    S2=(M**2).sum(0); S4=(M**4).sum(0); n=M.shape[0]
    tot2=S2.sum(); tot4=S4.sum()
    jk=[]
    for r in range(R):
        a2=(tot2-S2[r])/((R-1)*n); a4=(tot4-S4[r])/((R-1)*n)
        jk.append(1-a4/(3*a2**2))
    jk=np.array(jk)
    U=1-(tot4/(R*n))/(3*(tot2/(R*n))**2)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    # half comparison
    Uh1=U_of(M[:n//2]); Uh2=U_of(M[n//2:])
    # autocorrelation of m^2 averaged over replicas
    x=(M**2).mean(1); x=x-x.mean()
    ac=[np.dot(x[:-l],x[l:])/np.dot(x,x) for l in (1,5,10,20,50)]
    print(T,'U=%.5f err=%.5f'%(U,err),'halves %.4f %.4f'%(Uh1,Uh2),'acf(1,5,10,20,50)',np.round(ac,3))
