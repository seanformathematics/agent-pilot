
import sys, json, numpy as np
sys.path.insert(0,'/work')
from ana import jackknife_U, U_from_moments

def jackknife_general(S2,S4,n):
    S2=np.asarray(S2,float);S4=np.asarray(S4,float);n=np.asarray(n,float);R=len(S2)
    T2,T4,Nn=S2.sum(),S4.sum(),n.sum()
    U=U_from_moments(T2/Nn,T4/Nn)
    Ui=U_from_moments((T2-S2)/(Nn-n),(T4-S4)/(Nn-n))
    return U,np.sqrt((R-1)/R*np.sum((Ui-Ui.mean())**2))

Tlist=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
pts=[]
print("T    U            err_jk       chi2(3 late windows)  S=max(1,sqrt(chi2/2))   U_err_final    n_samples   N_eff_est(tau_int)")
for iT,T in enumerate(Tlist):
    d1=np.load(f"/work/res_{iT:02d}.npz");d2=np.load(f"/work/res2_{iT:02d}.npz");d3=np.load(f"/work/res3_{iT:02d}.npz")
    blk=100;nb=20;h=10
    L2=np.r_[d1['B2'][:,h:].sum(axis=1),d2['B2'][:,h:].sum(axis=1),d3['S2']]
    L4=np.r_[d1['B4'][:,h:].sum(axis=1),d2['B4'][:,h:].sum(axis=1),d3['S4']]
    Ln=np.r_[np.full(1000,1000),np.full(1000,1000),np.full(1000,2000)]
    U,err=jackknife_general(L2,L4,Ln)
    # three late windows
    A2=np.r_[d1['B2'][:,h:].sum(axis=1),d2['B2'][:,h:].sum(axis=1)];A4=np.r_[d1['B4'][:,h:].sum(axis=1),d2['B4'][:,h:].sum(axis=1)]
    wins=[(A2,A4,1000),(d3['B2'][:,:h].sum(axis=1),d3['B4'][:,:h].sum(axis=1),1000),(d3['B2'][:,h:].sum(axis=1),d3['B4'][:,h:].sum(axis=1),1000)]
    Us=[];es=[]
    for a2,a4,n in wins:
        u,e=jackknife_general(a2,a4,np.full(len(a2),n));Us.append(u);es.append(e)
    Us=np.array(Us);es=np.array(es);w=1/es**2;um=np.sum(w*Us)/w.sum()
    chi2=np.sum(((Us-um)/es)**2)
    S=max(1.0,np.sqrt(chi2/2))
    # tau_int from pooled ACF (reps 1-3 average) for N_eff info
    def tau_int(rho):
        tau=0.5
        for k in range(1,len(rho)):
            tau+=rho[k]
            if k>=6*tau: return tau
        return tau
    tau=tau_int((d1['acf2']+d2['acf2']+d3['acf2'])/3)
    nsamp=int(Ln.sum())
    uerr=float(err*S)
    pts.append(dict(T=T,status="ok",U=float(U),U_err=uerr,n_samples=nsamp))
    print(f"{T:.1f}  {U:.7f}  {err:.3e}   {chi2:5.2f}               {S:.3f}                {uerr:.3e}   {nsamp}   {nsamp/(2*tau):.2e} (tau={tau:.2f})")
print("max U_err = %.3e (target 5e-3)"%max(p['U_err'] for p in pts))
json.dump(pts,open('/work/final_points_submit.json','w'),indent=1)
print(json.dumps(pts))
# total simulation effort
print("sweeps*replicas total (production reps 1-3): %.2e"%(3*14*1000*(200+2000)+0))
