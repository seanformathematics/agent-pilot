
import sys, json, numpy as np
sys.path.insert(0,'/work')
from ana import jackknife_U, U_from_moments
from scipy import stats

def jackknife_general(S2,S4,n):
    """S2,S4: per-replica sums; n: per-replica counts (arrays). Estimator U(ratio of pooled means). Delete-one-replica jackknife."""
    S2=np.asarray(S2,float);S4=np.asarray(S4,float);n=np.asarray(n,float);R=len(S2)
    T2,T4,Nn=S2.sum(),S4.sum(),n.sum()
    U=U_from_moments(T2/Nn,T4/Nn)
    Ui=U_from_moments((T2-S2)/(Nn-n),(T4-S4)/(Nn-n))
    err=np.sqrt((R-1)/R*np.sum((Ui-Ui.mean())**2))
    Ubc=R*U-(R-1)*Ui.mean()
    return U,err,Ubc

Tlist=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
pts=[];chi2s=[];zall=[]
print("T    U_late        err_late     n_samples | window consistency chi2(2dof) | U_all(3 reps)  z(late-all) | U_rep3only")
for iT,T in enumerate(Tlist):
    d1=np.load(f"/work/res_{iT:02d}.npz");d2=np.load(f"/work/res2_{iT:02d}.npz");d3=np.load(f"/work/res3_{iT:02d}.npz")
    blk=int(d1['blk']);nb=d1['B2'].shape[1];h=nb//2
    # late data: second halves of reps 1,2 (sweeps 1201-2200); all of rep 3 (sweeps 1501-3500)
    L2=[];L4=[];Ln=[]
    for d in (d1,d2):
        L2.append(d['B2'][:,h:].sum(axis=1));L4.append(d['B4'][:,h:].sum(axis=1));Ln.append(np.full(d['B2'].shape[0],(nb-h)*blk))
    L2.append(d3['S2']);L4.append(d3['S4']);Ln.append(np.full(d3['S2'].shape[0],nb*blk))
    L2=np.concatenate(L2);L4=np.concatenate(L4);Ln=np.concatenate(Ln)
    U,err,Ubc=jackknife_general(L2,L4,Ln)
    nsamp=int(Ln.sum())
    # consistency of the three late windows: A=reps1,2 second halves, B=rep3 first half, C=rep3 second half
    A2=np.r_[d1['B2'][:,h:].sum(axis=1),d2['B2'][:,h:].sum(axis=1)];A4=np.r_[d1['B4'][:,h:].sum(axis=1),d2['B4'][:,h:].sum(axis=1)]
    UA,eA,_=jackknife_U(A2,A4,(nb-h)*blk)
    UB,eB,_=jackknife_U(d3['B2'][:,:h].sum(axis=1),d3['B4'][:,:h].sum(axis=1),h*blk)
    UC,eC,_=jackknife_U(d3['B2'][:,h:].sum(axis=1),d3['B4'][:,h:].sum(axis=1),(nb-h)*blk)
    Us=np.array([UA,UB,UC]);es=np.array([eA,eB,eC]);w=1/es**2;um=np.sum(w*Us)/w.sum()
    chi2=np.sum(((Us-um)/es)**2); chi2s.append(chi2)
    # all-data pooled (all 3 replicates, all post-200/1500 sweeps), per-replica equal n
    S2a=np.r_[d1['S2'],d2['S2'],d3['S2']];S4a=np.r_[d1['S4'],d2['S4'],d3['S4']]
    Ua,ea,_=jackknife_U(S2a,S4a,int(d1['nmeas']))
    U3,e3,_=jackknife_U(d3['S2'],d3['S4'],int(d3['nmeas']))
    # z of late vs all (they share data; use err_late as scale -> informal shift measure)
    zs=(U-Ua)/err; zall.append(zs)
    pts.append(dict(T=T,status="ok",U=float(U),U_err=float(err),n_samples=nsamp))
    print(f"{T:.1f}  {U:.7f}  {err:.3e}  {nsamp}  | chi2={chi2:.2f} | {Ua:.7f}  shift={zs:+.2f} err_late | {U3:.6f}")
chi2s=np.array(chi2s)
print("late-window consistency: total chi2=%.1f for 28 dof, p=%.3f"%(chi2s.sum(),1-stats.chi2.cdf(chi2s.sum(),28)))
print("shift late-vs-all in units of err_late: rms %.2f, max %.2f"%(np.sqrt(np.mean(np.array(zall)**2)),np.max(np.abs(zall))))
print("max U_err = %.3e"%max(p['U_err'] for p in pts))
json.dump(pts,open('/work/final_points_late.json','w'),indent=1)
print(json.dumps(pts))
