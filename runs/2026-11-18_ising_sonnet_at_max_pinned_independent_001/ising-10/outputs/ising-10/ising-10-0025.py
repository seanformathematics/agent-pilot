
import sys, numpy as np
sys.path.insert(0,'/work')
from ana import jackknife_U, U_from_moments
from scipy import stats

def jk_diff(Be2,Be4,Bl2,Bl4,ne,nl):
    R=len(Be2)
    f=lambda a2,a4,n,r: U_from_moments(a2/(r*n),a4/(r*n))
    full=f(Be2.sum(),Be4.sum(),ne,R)-f(Bl2.sum(),Bl4.sum(),nl,R)
    di=f(Be2.sum()-Be2,Be4.sum()-Be4,ne,R-1)-f(Bl2.sum()-Bl2,Bl4.sum()-Bl4,nl,R-1)
    return full,np.sqrt((R-1)/R*np.sum((di-di.mean())**2))

Tlist=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
Z3=[];ZEL3=[];ZEL12=[]
print("T    U3 (burn1500)        U12 (burn200)        z(3-12)  | early-late z: rep3  rep1+2 | U123 pooled")
rows=[]
for iT,T in enumerate(Tlist):
    d1=np.load(f"/work/res_{iT:02d}.npz");d2=np.load(f"/work/res2_{iT:02d}.npz");d3=np.load(f"/work/res3_{iT:02d}.npz")
    n=int(d1['nmeas']); blk=int(d1['blk'])
    assert int(d3['nmeas'])==n and int(d3['nburn'])==1500 and float(d3['T'])==T
    S2_12=np.r_[d1['S2'],d2['S2']];S4_12=np.r_[d1['S4'],d2['S4']]
    U12,e12,_=jackknife_U(S2_12,S4_12,n)
    U3,e3,_=jackknife_U(d3['S2'],d3['S4'],n)
    z=(U3-U12)/np.hypot(e3,e12); Z3.append(z)
    # early-late paired
    B2_3=d3['B2'];B4_3=d3['B4'];nb=B2_3.shape[1];h=nb//2
    f,e=jk_diff(B2_3[:,:h].sum(axis=1),B4_3[:,:h].sum(axis=1),B2_3[:,h:].sum(axis=1),B4_3[:,h:].sum(axis=1),h*blk,(nb-h)*blk); zel3=f/e
    B2_12=np.vstack([d1['B2'],d2['B2']]);B4_12=np.vstack([d1['B4'],d2['B4']])
    f,e=jk_diff(B2_12[:,:h].sum(axis=1),B4_12[:,:h].sum(axis=1),B2_12[:,h:].sum(axis=1),B4_12[:,h:].sum(axis=1),h*blk,(nb-h)*blk); zel12=f/e
    ZEL3.append(zel3); ZEL12.append(zel12)
    S2=np.r_[S2_12,d3['S2']];S4=np.r_[S4_12,d3['S4']]
    U,err,_=jackknife_U(S2,S4,n)
    rows.append((T,U,err))
    print(f"{T:.1f}  {U3:.6f}+-{e3:.6f}  {U12:.6f}+-{e12:.6f}  {z:+.2f}   |   {zel3:+.2f}   {zel12:+.2f}   | {U:.6f}+-{err:.6f}")
Z3=np.array(Z3);ZEL3=np.array(ZEL3);ZEL12=np.array(ZEL12)
print("z(3-12): mean %.2f (+-%.2f), rms %.2f, chi2=%.1f/14 p=%.3f"%(Z3.mean(),1/np.sqrt(14),np.sqrt(np.mean(Z3**2)),np.sum(Z3**2),1-stats.chi2.cdf(np.sum(Z3**2),14)))
print("early-late z rep3 (burn 1500): rms %.2f, chi2=%.1f/14 p=%.3f; mean %.2f"%(np.sqrt(np.mean(ZEL3**2)),np.sum(ZEL3**2),1-stats.chi2.cdf(np.sum(ZEL3**2),14),ZEL3.mean()))
print("early-late z rep1+2 (burn 200): rms %.2f, chi2=%.1f/14 p=%.3f"%(np.sqrt(np.mean(ZEL12**2)),np.sum(ZEL12**2),1-stats.chi2.cdf(np.sum(ZEL12**2),14)))
print("corr(z_el rep3, z_el rep1+2) = %.2f"%np.corrcoef(ZEL3,ZEL12)[0,1])
np.save('/work/pooled123.npy',np.array(rows))

# absolute-time history of U at T=2.2 and T=1.5 across all runs
for iT,T in [(7,2.2),(0,1.5)]:
    print(f"--- U history at T={T}: (run, window in absolute sweeps) U +- err")
    for pref,burn in [('res',200),('res2',200),('res3',1500)]:
        d=np.load(f"/work/{pref}_{iT:02d}.npz");B2=d['B2'];B4=d['B4'];blk=int(d['blk']);nb=B2.shape[1];h=nb//2
        for (a,b) in [(0,h),(h,nb)]:
            U,e,_=jackknife_U(B2[:,a:b].sum(axis=1),B4[:,a:b].sum(axis=1),(b-a)*blk)
            print(f"   {pref}: [{burn+a*blk},{burn+b*blk})  U={U:.6f}+-{e:.6f}  (R={B2.shape[0]})")
