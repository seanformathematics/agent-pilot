
import sys, numpy as np
sys.path.insert(0,'/work')
from ana import U_from_moments
from scipy import stats

def jk_diff(Be2,Be4,Bl2,Bl4,ne,nl):
    """delete-one-replica jackknife of U_early - U_late. B*: per-replica sums."""
    R=len(Be2)
    def Uf(a2,a4,n,Rr): return U_from_moments(a2/(Rr*n),a4/(Rr*n))
    full=Uf(Be2.sum(),Be4.sum(),ne,R)-Uf(Bl2.sum(),Bl4.sum(),nl,R)
    Ue=Uf(Be2.sum()-Be2,Be4.sum()-Be4,ne,R-1); Ul=Uf(Bl2.sum()-Bl2,Bl4.sum()-Bl4,nl,R-1)
    di=Ue-Ul
    err=np.sqrt((R-1)/R*np.sum((di-di.mean())**2))
    return full,err

Tlist=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
Z1=[];Z2=[];Zp=[];chis=[]
print("T    z_early-late(paired U): rep1  rep2  pooled | chi2(20 blocks, pooled U-blocks) p | chi2 of m2 block means p")
for iT,T in enumerate(Tlist):
    zz=[]
    Ball2=[];Ball4=[]
    for pref in ('res','res2'):
        d=np.load(f"/work/{pref}_{iT:02d}.npz"); B2=d['B2'];B4=d['B4']; blk=int(d['blk']); nb=B2.shape[1]; h=nb//2
        f,e=jk_diff(B2[:,:h].sum(axis=1),B4[:,:h].sum(axis=1),B2[:,h:].sum(axis=1),B4[:,h:].sum(axis=1),h*blk,(nb-h)*blk)
        zz.append(f/e); Ball2.append(B2); Ball4.append(B4)
    B2=np.vstack(Ball2);B4=np.vstack(Ball4)
    f,e=jk_diff(B2[:,:h].sum(axis=1),B4[:,:h].sum(axis=1),B2[:,h:].sum(axis=1),B4[:,h:].sum(axis=1),h*blk,(nb-h)*blk)
    zp=f/e
    # chi2 of m2 block means across the 20 blocks using replica-based covariance (full covariance of block means)
    X=B2/blk; R=X.shape[0]
    mu=X.mean(axis=0); C=np.cov(X.T)/R
    dmu=mu-mu.mean()
    # project onto (nb-1)-dim subspace orthogonal to constant
    Pm=np.eye(nb)-np.ones((nb,nb))/nb
    Cp=Pm@C@Pm
    w,v=np.linalg.eigh(Cp); keep=w>1e-12*w.max()
    chi2=np.sum((v[:,keep].T@dmu)**2/w[keep]); dof=keep.sum()
    p=1-stats.chi2.cdf(chi2,dof)
    Z1.append(zz[0]);Z2.append(zz[1]);Zp.append(zp);chis.append((chi2,dof,p))
    print(f"{T:.1f}   {zz[0]:+.2f}  {zz[1]:+.2f}  {zp:+.2f}   | chi2(m2 blocks full-cov)={chi2:.1f}/{dof} p={p:.3f}")
Z1=np.array(Z1);Z2=np.array(Z2);Zp=np.array(Zp)
print("rms z (paired): rep1 %.2f rep2 %.2f pooled %.2f ; mean z pooled %.2f"%(np.sqrt(np.mean(Z1**2)),np.sqrt(np.mean(Z2**2)),np.sqrt(np.mean(Zp**2)),Zp.mean()))
ps=np.array([c[2] for c in chis]); print("p-values of chi2 tests: min %.3f; KS test uniformity p=%.3f"%(ps.min(),stats.kstest(ps,'uniform').pvalue))
