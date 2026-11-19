
import sys, numpy as np
sys.path.insert(0,'/work')
from ana import jackknife_U, U_from_moments
from exact import exact_e_C

N=16; nsite=N*N
out=[]
print("T     U         err_jk(rep)  err_jk(blk)  err_boot   | hot-cold z | early-late z | e z    C_mc/C_ex   | tau2  tau4  Neff(m2) | burn-in last100 hot/cold vs prod <m2>")
zs_hc=[]; zs_el=[]; zs_e=[]; zs_C=[]
rngb=np.random.default_rng(99)
for iT in range(14):
    d=np.load(f"/work/res_{iT:02d}.npz")
    T=float(d['T']); R=int(d['R']); Rh=int(d['Rh']); nmeas=int(d['nmeas']); blk=int(d['blk'])
    S2,S4,B2,B4=d['S2'],d['S4'],d['B2'],d['B4']
    U,err,Ubc=jackknife_U(S2,S4,nmeas)
    # block jackknife: units = individual blocks (R*nb), each with blk measurements
    b2=B2.reshape(-1); b4=B4.reshape(-1)
    Ub,errb,_=jackknife_U(b2,b4,blk)
    # bootstrap over replicas
    nb=2000; Ubs=np.empty(nb)
    for k in range(nb):
        ii=rngb.integers(0,R,R)
        Ubs[k]=U_from_moments(S2[ii].sum()/(R*nmeas),S4[ii].sum()/(R*nmeas))
    errboot=Ubs.std(ddof=1)
    # hot vs cold
    Uh,eh,_=jackknife_U(S2[:Rh],S4[:Rh],nmeas); Uc,ec,_=jackknife_U(S2[Rh:],S4[Rh:],nmeas)
    zhc=(Uh-Uc)/np.hypot(eh,ec)
    # early vs late halves
    nbk=B2.shape[1]; h=nbk//2
    Ue,ee_,_=jackknife_U(B2[:,:h].sum(axis=1),B4[:,:h].sum(axis=1),h*blk)
    Ul,el_,_=jackknife_U(B2[:,h:].sum(axis=1),B4[:,h:].sum(axis=1),(nbk-h)*blk)
    zel=(Ue-Ul)/np.hypot(ee_,el_)
    # energy
    nE=int(d['nE']); e_r=d['SE']/nE/nsite
    e_mc=e_r.mean(); e_err=e_r.std(ddof=1)/np.sqrt(R)
    e_ex,C_ex=exact_e_C(N,T)
    ze=(e_mc-e_ex)/e_err
    E1=d['SE'].sum()/(R*nE); E2=d['SE2'].sum()/(R*nE)
    C_mc=(E2-E1**2)/T**2/nsite
    # tau
    def tau_int(rho):
        tau=0.5
        for k in range(1,len(rho)):
            tau+=rho[k]
            if k>=6*tau: return tau
        return tau
    t2=tau_int(d['acf2']); t4=tau_int(d['acf4'])
    Neff=R*nmeas/(2*t2)
    burn=d['burn']; bl=burn[-100:].mean(axis=0); prod=S2.sum()/(R*nmeas)
    zs_hc.append(zhc); zs_el.append(zel); zs_e.append(ze)
    print(f"{T:.1f}  {U:.6f}  {err:.6f}    {errb:.6f}    {errboot:.6f}  | {zhc:+.2f}  | {zel:+.2f} | {ze:+.2f}  {C_mc/C_ex:.4f} | {t2:.2f} {t4:.2f} {Neff:.2e} | {bl[0]:.4f}/{bl[1]:.4f} vs {prod:.4f}")
    out.append((T,U,err,errb,errboot,R*nmeas))
np.save('/work/summary.npy',np.array(out))
zs_hc=np.array(zs_hc); zs_el=np.array(zs_el); zs_e=np.array(zs_e)
print("z-score summaries (should be ~N(0,1)): hot-cold mean %.2f rms %.2f | early-late mean %.2f rms %.2f | energy mean %.2f rms %.2f"%(zs_hc.mean(),np.sqrt((zs_hc**2).mean()),zs_el.mean(),np.sqrt((zs_el**2).mean()),zs_e.mean(),np.sqrt((zs_e**2).mean())))
