
import numpy as np, sys, time, json
sys.path.insert(0,'/work')
import ising
def U_jack(m):
    m2 = m**2; m4 = m**4
    n = m.shape[0]
    A = m2.sum(axis=0); B = m4.sum(axis=0)
    Atot = A.sum(); Btot = B.sum(); ntot = n*m.shape[1]
    U = 1 - (Btot/ntot)/(3*(Atot/ntot)**2)
    R = m.shape[1]
    nr = ntot - n
    a = (Atot - A)/nr; b = (Btot - B)/nr
    Ur = 1 - b/(3*a*a)
    err = np.sqrt((R-1)/R * np.sum((Ur - Ur.mean())**2))
    return U, err
temps = [1.5,1.6,1.7,1.8,1.9,2.0,2.1]
seeds = [1001, 2002]
out = {}
t0=time.time()
for T in temps:
    parts=[]; Us=[]; es=[]
    for sd in seeds:
        S,m,_ = ising.run_chain(T, 1000, 2000, seed=sd+int(round(T*100)), nburn=500, rec_E=False, N=16)
        parts.append(m)
        U,e = U_jack(m); Us.append(U); es.append(e)
    M = np.concatenate(parts, axis=1)
    U,e = U_jack(M)
    np.save('/work/m_T%.2f.npy'%T, M.astype(np.float32))
    out[T] = dict(U=U, err=e, U_seed1=Us[0], U_seed2=Us[1], e1=es[0], e2=es[1])
    print("T=%.2f U=%.5f err=%.5f | seedA U=%.5f(%.5f) seedB U=%.5f(%.5f) | t=%.0fs"%(T,U,e,Us[0],es[0],Us[1],es[1],time.time()-t0), flush=True)
json.dump({str(k):v for k,v in out.items()}, open('/work/prod1.json','w'), indent=1)
