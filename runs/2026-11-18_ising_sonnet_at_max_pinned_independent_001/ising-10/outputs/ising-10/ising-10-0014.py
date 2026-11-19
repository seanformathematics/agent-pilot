
import sys, numpy as np
sys.path.insert(0,'/work')
from ana import jackknife_U
from scipy import stats
zs=[]
rows=[]
print("T     U1          err1        U2          err2        diff      z      | pooled U     pooled err (2000 replicas)")
for iT in range(14):
    d1=np.load(f"/work/res_{iT:02d}.npz"); d2=np.load(f"/work/res2_{iT:02d}.npz")
    n=int(d1['nmeas']); assert n==int(d2['nmeas'])
    U1,e1,_=jackknife_U(d1['S2'],d1['S4'],n); U2,e2,_=jackknife_U(d2['S2'],d2['S4'],n)
    z=(U1-U2)/np.hypot(e1,e2); zs.append(z)
    Up,ep,Ubc=jackknife_U(np.concatenate([d1['S2'],d2['S2']]),np.concatenate([d1['S4'],d2['S4']]),n)
    rows.append((float(d1['T']),Up,ep))
    print(f"{float(d1['T']):.1f}  {U1:.6f}  {e1:.6f}  {U2:.6f}  {e2:.6f}  {U1-U2:+.6f}  {z:+.2f}  | {Up:.6f}  {ep:.6f}")
zs=np.array(zs)
chi2=np.sum(zs**2)
print("mean z=%.2f (+-%.2f), rms z=%.2f, chi2=%.1f for 14 dof, p=%.3f"%(zs.mean(),1/np.sqrt(14),np.sqrt(np.mean(zs**2)),chi2,1-stats.chi2.cdf(chi2,14)))
np.save('/work/pooled12.npy',np.array(rows))
