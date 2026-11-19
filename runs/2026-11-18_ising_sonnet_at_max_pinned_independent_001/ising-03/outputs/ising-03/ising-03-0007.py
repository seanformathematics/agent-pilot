
import sys, pickle
sys.path.insert(0,'/work')
import numpy as np
import sw, exact
res=pickle.load(open('/work/pilot16.pkl','rb'))
for T in [1.9, 2.5, 2.4, 1.5]:
    r=res[T]; eb=r['eblk']; R=r['R']; half=R//2
    ee=exact.energy_per_site_exact(T,16)
    print(f"T={T} exact e={ee:.6f}")
    for name,sl in (("hot half",slice(0,half)),("cold half",slice(half,R)),("all",slice(0,R))):
        x=eb[sl].mean(axis=1)
        print(f"   {name}: e={x.mean():.6f} +- {x.std(ddof=1)/np.sqrt(len(x)):.6f}   z={(x.mean()-ee)/(x.std(ddof=1)/np.sqrt(len(x))):.2f}")
    tb=eb.mean(axis=0); 
    print("   block means (over replicas) minus exact (x1e3):", ((tb-ee)*1e3).round(2))
    x=eb.mean(axis=1)
    print("   replica-level: skew-ish: max-min=%.4f, sorted top5 deviations x1e3:"%(x.max()-x.min()), ((np.sort(x)[-5:]-ee)*1e3).round(2), "median-ee (x1e3) %.3f"%((np.median(x)-ee)*1e3), "std", x.std(ddof=1).round(5))
    print("   burn-in traj hot[:10]:", r['traj_hot'][:10].round(3), " cold[:10]:", r['traj_cold'][:10].round(3))
