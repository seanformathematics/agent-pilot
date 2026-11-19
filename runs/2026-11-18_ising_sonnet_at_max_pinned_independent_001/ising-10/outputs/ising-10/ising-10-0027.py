
import sys, numpy as np
sys.path.insert(0,'/work')
from ana import jackknife_U, U_from_moments
from scipy import stats

def jackknife_general(S2,S4,n):
    S2=np.asarray(S2,float);S4=np.asarray(S4,float);n=np.asarray(n,float);R=len(S2)
    T2,T4,Nn=S2.sum(),S4.sum(),n.sum()
    U=U_from_moments(T2/Nn,T4/Nn)
    Ui=U_from_moments((T2-S2)/(Nn-n),(T4-S4)/(Nn-n))
    return U,np.sqrt((R-1)/R*np.sum((Ui-Ui.mean())**2))

for iT,T,ir in [(7,2.2,0),(0,1.5,1)]:
    d1=np.load(f"/work/res_{iT:02d}.npz");d2=np.load(f"/work/res2_{iT:02d}.npz");d3=np.load(f"/work/res3_{iT:02d}.npz");dr=np.load(f"/work/relax_{ir}.npz")
    blk=100;h=10
    wins={}
    wins['rep12 W2 [1200,2200)']=(np.r_[d1['B2'][:,h:].sum(axis=1),d2['B2'][:,h:].sum(axis=1)],np.r_[d1['B4'][:,h:].sum(axis=1),d2['B4'][:,h:].sum(axis=1)],1000)
    wins['rep3 [1500,2500)']=(d3['B2'][:,:h].sum(axis=1),d3['B4'][:,:h].sum(axis=1),1000)
    wins['rep3 [2500,3500)']=(d3['B2'][:,h:].sum(axis=1),d3['B4'][:,h:].sum(axis=1),1000)
    wins['relax [1200,2200)']=(dr['B2'][:,12:22].sum(axis=1),dr['B4'][:,12:22].sum(axis=1),1000)
    wins['relax [2200,3200)']=(dr['B2'][:,22:32].sum(axis=1),dr['B4'][:,22:32].sum(axis=1),1000)
    wins['relax [3200,4000)']=(dr['B2'][:,32:40].sum(axis=1),dr['B4'][:,32:40].sum(axis=1),800)
    print(f"=== T={T}")
    Us=[];es=[]
    for k,(a2,a4,n) in wins.items():
        U,e=jackknife_general(a2,a4,np.full(len(a2),n)); Us.append(U);es.append(e)
        print(f"   {k:22s} U={U:.7f} +- {e:.7f}  (R={len(a2)}, n={n})")
    Us=np.array(Us);es=np.array(es);w=1/es**2;um=np.sum(w*Us)/w.sum()
    chi2=np.sum(((Us-um)/es)**2);dof=len(Us)-1
    print(f"   weighted mean {um:.7f} +- {1/np.sqrt(w.sum()):.7f};  chi2={chi2:.2f}/{dof}  p={1-stats.chi2.cdf(chi2,dof):.3f}")
    # pooled generalized jackknife over all replicas of all these windows (replicas of the same run kept as units with both halves summed)
    # production late set (reps1,2 second halves + rep3 whole)
    P2=np.r_[d1['B2'][:,h:].sum(axis=1),d2['B2'][:,h:].sum(axis=1),d3['S2']];P4=np.r_[d1['B4'][:,h:].sum(axis=1),d2['B4'][:,h:].sum(axis=1),d3['S4']]
    Pn=np.r_[np.full(1000,1000),np.full(1000,1000),np.full(1000,2000)]
    Up,ep=jackknife_general(P2,P4,Pn)
    # production-late + relax late [1200,4000)
    R2=dr['B2'][:,12:].sum(axis=1);R4=dr['B4'][:,12:].sum(axis=1);Rn=np.full(len(R2),2800)
    Ua,ea=jackknife_general(np.r_[P2,R2],np.r_[P4,R4],np.r_[Pn,Rn])
    print(f"   production-late only: U={Up:.7f}+-{ep:.7f} (n={Pn.sum()});  production-late + relax-late: U={Ua:.7f}+-{ea:.7f} (n={Pn.sum()+Rn.sum()})")
