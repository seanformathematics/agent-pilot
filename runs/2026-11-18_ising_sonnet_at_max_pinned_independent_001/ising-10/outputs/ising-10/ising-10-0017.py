
import sys, json, numpy as np
sys.path.insert(0,'/work')
from ana import jackknife_U, U_from_moments
from exact import exact_e_C
from scipy import stats

N=16; nsite=N*N
Tlist=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def tau_int(rho):
    tau=0.5
    for k in range(1,len(rho)):
        tau+=rho[k]
        if k>=6*tau: return tau
    return tau
pts=[]; zhc=[]; zel=[]; ze=[]
print("T    U            U_err        n_samples | U_bc-U     blk-jk err | hot-cold z, early-late z, e z | tau_int(m2)  tau_int(m4)")
for iT,T in enumerate(Tlist):
    d1=np.load(f"/work/res_{iT:02d}.npz"); d2=np.load(f"/work/res2_{iT:02d}.npz")
    assert float(d1['T'])==T and float(d2['T'])==T
    n=int(d1['nmeas']); R1=int(d1['R']); Rh=int(d1['Rh']); blk=int(d1['blk'])
    S2=np.concatenate([d1['S2'],d2['S2']]); S4=np.concatenate([d1['S4'],d2['S4']])
    B2=np.vstack([d1['B2'],d2['B2']]); B4=np.vstack([d1['B4'],d2['B4']])
    R=len(S2)
    U,err,Ubc=jackknife_U(S2,S4,n)
    _,errblk,_=jackknife_U(B2.reshape(-1),B4.reshape(-1),blk)
    # hot (first Rh of each replicate) vs cold (rest)
    hot=np.r_[np.arange(Rh),R1+np.arange(Rh)]; cold=np.setdiff1d(np.arange(R),hot)
    Uh,eh,_=jackknife_U(S2[hot],S4[hot],n); Uc,ec,_=jackknife_U(S2[cold],S4[cold],n)
    nb=B2.shape[1]; h=nb//2
    Ue,ee,_=jackknife_U(B2[:,:h].sum(axis=1),B4[:,:h].sum(axis=1),h*blk)
    Ul,el,_=jackknife_U(B2[:,h:].sum(axis=1),B4[:,h:].sum(axis=1),(nb-h)*blk)
    nE=int(d1['nE']); e_r=np.concatenate([d1['SE'],d2['SE']])/nE/nsite
    e_ex,_=exact_e_C(N,T)
    z_e=(e_r.mean()-e_ex)/(e_r.std(ddof=1)/np.sqrt(R))
    z_hc=(Uh-Uc)/np.hypot(eh,ec); z_el=(Ue-Ul)/np.hypot(ee,el)
    t2=tau_int(0.5*(d1['acf2']+d2['acf2'])); t4=tau_int(0.5*(d1['acf4']+d2['acf4']))
    zhc.append(z_hc); zel.append(z_el); ze.append(z_e)
    pts.append(dict(T=T,status="ok",U=float(U),U_err=float(err),n_samples=int(R*n)))
    print(f"{T:.1f}  {U:.7f}  {err:.3e}  {R*n:d} | {Ubc-U:+.1e}  {errblk:.3e} | {z_hc:+.2f} {z_el:+.2f} {z_e:+.2f} | {t2:.2f} {t4:.2f}")
zhc=np.array(zhc); zel=np.array(zel); ze=np.array(ze)
print("rms z: hot-cold %.2f, early-late %.2f, energy %.2f (n=14 each; expect ~1)"%tuple(np.sqrt(np.mean(a**2)) for a in (zhc,zel,ze)))
print("max U_err = %.3e (target 5e-3)"%max(p['U_err'] for p in pts))
json.dump(pts,open('/work/final_points.json','w'),indent=1)
print(json.dumps(pts))
# seed lists
seeds_prod=[880000+i for i in range(14)]+[990000+i for i in range(14)]
seeds_pilot=[2024000+i for i in range(14)]
seeds_metro=[5150+i for i in range(3)]+[616100+i for i in range(3)]
seeds_n8=[424200+i for i in range(3)]+[717100+i for i in range(5)]
seeds_n4=[777+int(T*100) for T in [1.5,2.0,2.5,3.0,4.0]]+[31000+int(T*1000)+b for T in [2.269,2.5,3.0] for b in range(12)]
print("prod seeds:",seeds_prod)
print("other seeds:",seeds_pilot+seeds_metro+seeds_n8+seeds_n4+[99,12345])
