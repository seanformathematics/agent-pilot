
import numpy as np, sys, time
sys.path.insert(0,'/work')
import ising
def U_jack(m):
    # m: (nsweep, R) array; jackknife over replicas (columns)
    m2 = m**2; m4 = m**4
    n = m.shape[0]
    A = m2.sum(axis=0); B = m4.sum(axis=0)   # per replica sums
    Atot = A.sum(); Btot = B.sum(); ntot = n*m.shape[1]
    U = 1 - (Btot/ntot)/(3*(Atot/ntot)**2)
    R = m.shape[1]
    Ur = np.empty(R)
    for r in range(R):
        nr = ntot - n
        a = (Atot - A[r])/nr; b = (Btot - B[r])/nr
        Ur[r] = 1 - b/(3*a*a)
    err = np.sqrt((R-1)/R * np.sum((Ur - Ur.mean())**2))
    return U, err, Ur.mean()
t0=time.time()
S,m,E = ising.run_chain(2.3, 1000, 1000, seed=101, nburn=300, rec_E=False, N=16)
print("time", time.time()-t0)
U, err, Um = U_jack(m[:, :])
print("T=2.3 R=1000 nsweep=1000: U=%.5f err=%.5f jackmean=%.5f"%(U,err,Um))
# check error scaling: use subsets of replicas
for R in [125, 250, 500]:
    errs=[]
    for k in range(1000//R):
        _,e,_ = U_jack(m[:, k*R:(k+1)*R]); errs.append(e)
    print("R=%d: mean subset err %.5f"%(R, np.mean(errs)))
# error scaling with sweeps: use first 250 sweeps
_,e,_ = U_jack(m[:250]); print("nsweep=250, R=1000 err %.5f"%e)
_,e,_ = U_jack(m[:500]); print("nsweep=500, R=1000 err %.5f"%e)
