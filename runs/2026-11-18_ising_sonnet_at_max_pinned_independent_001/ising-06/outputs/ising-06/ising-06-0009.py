
import sys
sys.path.insert(0,'/work')
import numpy as np, run, exact_torus as ex
N=16
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
rng=np.random.default_rng(424242)

def U_pool(X, i2, i4, sel=None):
    A=X[i2].mean(axis=1); Bq=X[i4].mean(axis=1)
    if sel is not None: A=A[sel]; Bq=Bq[sel]
    return run.U_from(A.mean(), Bq.mean())

summary={}
chi_e=[]; z_e=[]; zs_ab=[]; zs_hc=[]; zs_pi=[]
print(" T    U_imp    err_rep  err_blk  err_boot | U_plain  err   | z(plain-imp) | e-e_exact z | z(A-B) z(hot-cold)")
rows=[]
for T in Ts:
    d=np.load(f"/work/prod_T{T:.1f}.npz")
    X=d['X']; nrep=X.shape[1]; nb=X.shape[2]
    Ui,Uie=run.jack_U(X,2,3)
    Up,Upe=run.jack_U(X,0,1)
    # block-level jackknife (units = replica x block), leave-one-unit-out via grouped delete (groups of 10 blocks) for speed
    a=X[2].reshape(-1); b=X[3].reshape(-1)    # all blocks, equal weights
    # use delete-group jackknife with g groups of consecutive blocks within each replica
    g=nrep*8
    ag=a.reshape(g,-1).mean(axis=1); bg=b.reshape(g,-1).mean(axis=1)
    sA,sB=ag.sum(),bg.sum()
    Uj=run.U_from((sA-ag)/(g-1),(sB-bg)/(g-1))
    err_blk=np.sqrt((g-1)/g*np.sum((Uj-Uj.mean())**2))
    # bootstrap over replicas
    A=X[2].mean(axis=1); Bq=X[3].mean(axis=1)
    bs=[]
    for _ in range(1000):
        idx=rng.integers(0,nrep,nrep)
        bs.append(run.U_from(A[idx].mean(),Bq[idx].mean()))
    err_boot=np.std(bs,ddof=1)
    # difference plain - improved by jackknife over replicas
    P2=X[0].mean(axis=1); P4=X[1].mean(axis=1)
    sP2,sP4,sA_,sB_=P2.sum(),P4.sum(),A.sum(),Bq.sum()
    dj=run.U_from((sP2-P2)/(nrep-1),(sP4-P4)/(nrep-1)) - run.U_from((sA_-A)/(nrep-1),(sB_-Bq)/(nrep-1))
    d_full=Up-Ui
    d_err=np.sqrt((nrep-1)/nrep*np.sum((dj-dj.mean())**2))
    z_pi=d_full/d_err if d_err>0 else 0.0
    # energy vs exact
    e,ee=run.jack_mean(X[4].mean(axis=1)); eex=ex.energy_per_spin_exact(T,N)
    z_e.append((e-eex)/ee)
    # halves A (batches 0,1) vs B (batches 2,3) ; hot vs cold
    R=128
    selA=np.r_[0:2*R]; selB=np.r_[2*R:4*R]
    selH=np.r_[0:R, 2*R:3*R]; selC=np.r_[R:2*R, 3*R:4*R]
    def sub_err(sel):
        Xs=X[:,sel,:]
        return run.jack_U(Xs,2,3)
    UA,eA=sub_err(selA); UB,eB=sub_err(selB); UH,eH=sub_err(selH); UC,eC=sub_err(selC)
    z_ab=(UA-UB)/np.hypot(eA,eB); z_hc=(UH-UC)/np.hypot(eH,eC)
    zs_ab.append(z_ab); zs_hc.append(z_hc); zs_pi.append(z_pi)
    print(f"{T:.1f} {Ui:.5f} {Uie:.5f} {err_blk:.5f} {err_boot:.5f} | {Up:.5f} {Upe:.5f} | {z_pi:+.2f} | {(e-eex)/ee:+.2f} | {z_ab:+.2f} {z_hc:+.2f}")
    # per-batch calibration: 4 batches of 128 replicas
    ub=[];eb=[]
    for k in range(4):
        sel=np.r_[k*R:(k+1)*R]
        u,er=run.jack_U(X[:,sel,:],2,3); ub.append(u); eb.append(er)
    ub=np.array(ub); eb=np.array(eb)
    rows.append((T,Ui,Uie,ub,eb))

print()
z_e=np.array(z_e)
print("energy z-scores:", np.round(z_e,2), " chi2=%.1f for 14 dof"%np.sum(z_e**2))
print("A-B z-scores   : chi2=%.1f for 14 dof"%np.sum(np.array(zs_ab)**2))
print("hot-cold z     : chi2=%.1f for 14 dof"%np.sum(np.array(zs_hc)**2))
print("plain-imp z    : chi2=%.1f for 14 dof"%np.sum(np.array(zs_pi)**2))
# per-batch calibration: chi2 about the T-wise weighted mean of 4 batches
chi=0; dof=0; ratios=[]
for T,Ui,Uie,ub,eb in rows:
    w=1/eb**2; mu=np.sum(w*ub)/np.sum(w)
    chi+=np.sum(w*(ub-mu)**2); dof+=3
    ratios.append(np.std(ub,ddof=1)/np.mean(eb))
print("per-batch calibration: chi2=%.1f for %d dof (expect %d +- %.1f)"%(chi,dof,dof,np.sqrt(2*dof)))
print("ratio std(U_batch)/mean(err_batch) per T:", np.round(ratios,2))
