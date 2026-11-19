
import sys, numpy as np
sys.path.insert(0,'/work')
from ana import jackknife_U
np.set_printoptions(linewidth=200, precision=4, suppress=True)
for iT,T in [(7,2.2),(0,1.5),(4,1.9)]:
    print("=== T =",T)
    for rep,pref in [(1,'res'),(2,'res2')]:
        d=np.load(f"/work/{pref}_{iT:02d}.npz")
        Rh=int(d['Rh']); blk=int(d['blk']); B2=d['B2']/blk; B4=d['B4']/blk
        nb=B2.shape[1]
        for name,sl in [('hot',slice(0,Rh)),('cold',slice(Rh,None))]:
            X=B2[sl]; mu=X.mean(axis=0); se=X.std(axis=0,ddof=1)/np.sqrt(X.shape[0])
            # early (first 5 blocks) vs late (last 5 blocks)
            e=X[:,:5].mean(axis=1); l=X[:,-5:].mean(axis=1)
            diff=(e-l); dm=diff.mean(); dse=diff.std(ddof=1)/np.sqrt(len(diff))   # paired difference per replica (accounts for correlation)
            print(f" rep{rep} {name}: m2 block-mean (x1e3 dev from overall mean) first/last blocks:", np.round((mu[[0,1,2,3,16,17,18,19]]-mu.mean())*1e3,2), f"| per-block se ~{se.mean()*1e3:.2f}e-3 | paired early-late m2 diff = {dm*1e3:+.2f}e-3 +- {dse*1e3:.2f}e-3 (z={dm/dse:+.2f})")
    # combined paired test on m2, m4 and U-related quantities across all 2000 replicas
    B2=np.vstack([np.load(f"/work/res_{iT:02d}.npz")['B2'],np.load(f"/work/res2_{iT:02d}.npz")['B2']])/100
    B4=np.vstack([np.load(f"/work/res_{iT:02d}.npz")['B4'],np.load(f"/work/res2_{iT:02d}.npz")['B4']])/100
    for nm,X in [('m2',B2),('m4',B4)]:
        diff=X[:,:10].mean(axis=1)-X[:,10:].mean(axis=1)
        print(f" pooled paired first-half minus second-half {nm}: {diff.mean():+.3e} +- {diff.std(ddof=1)/np.sqrt(len(diff)):.3e}  z={diff.mean()/(diff.std(ddof=1)/np.sqrt(len(diff))):+.2f}")
