
import sys, numpy as np
sys.path.insert(0,'/work')
from ana import jackknife_U
from scipy import stats
print("T    chi2(m2 blocks, 19dof) p    chi2(m4) p    slope(m2 per 1000 sweeps)+-err   | U quarters (+-err)")
for iT in range(14):
    d=np.load(f"/work/res_{iT:02d}.npz")
    T=float(d['T']); R=int(d['R']); blk=int(d['blk']); B2=d['B2']/blk; B4=d['B4']/blk   # (R, nb) block means
    nb=B2.shape[1]
    res=[]
    for B in (B2,B4):
        mu=B.mean(axis=0)                       # block mean over replicas
        # covariance across replicas of the nb block means (accounts for all structure)
        se=B.std(axis=0,ddof=1)/np.sqrt(R)
        chi2=np.sum(((mu-mu.mean())/se)**2)    # blocks assumed independent across (replicas independent, blocks nearly so)
        res.append((chi2,1-stats.chi2.cdf(chi2,nb-1)))
    # slope of m2 block means
    mu=B2.mean(axis=0); se=B2.std(axis=0,ddof=1)/np.sqrt(R)
    x=(np.arange(nb)*blk+blk/2)/1000.
    w=1/se**2; X=np.vstack([np.ones(nb),x]).T
    cov=np.linalg.inv(X.T@(X*w[:,None])); beta=cov@(X.T@(w*mu))
    # quarters
    q=nb//4; Uq=[]
    for k in range(4):
        sl=slice(k*q,(k+1)*q)
        U,e,_=jackknife_U(d['B2'][:,sl].sum(axis=1),d['B4'][:,sl].sum(axis=1),q*blk)
        Uq.append(f"{U:.4f}+-{e:.4f}")
    print(f"{T:.1f}  {res[0][0]:6.1f} {res[0][1]:.3f}      {res[1][0]:6.1f} {res[1][1]:.3f}     {beta[1]:+.5f}+-{np.sqrt(cov[1,1]):.5f}   | "+"  ".join(Uq))
