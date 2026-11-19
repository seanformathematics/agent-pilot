
import sys, time, os
sys.path.insert(0,'/work')
import numpy as np, run, exact_torus as ex

N=16
ALL_T=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
todo=[1.5,1.6,1.7,1.8,1.9,2.0,2.1]
R=128; n_eq=300; n_meas=4000; B=50
starts=["hot","cold","hot","cold"]
t00=time.time()
for T in todo:
    fn=f"/work/prod_T{T:.1f}.npz"
    if os.path.exists(fn):
        print("skip", T); continue
    iT=ALL_T.index(T)
    seeds=[100000+1000*iT+k for k in range(4)]
    Xs=[]
    t0=time.time()
    for k,(sd,st) in enumerate(zip(seeds,starts)):
        X,_=run.run_T(N,T,R=R,n_eq=n_eq,n_meas=n_meas,B=B,seed=sd,start=st)
        Xs.append(X)
    X=np.concatenate(Xs,axis=1)   # (5, 512, 80)
    np.savez(fn, X=X, seeds=np.array(seeds), T=T, R=R, n_eq=n_eq, n_meas=n_meas, B=B, starts=np.array(starts))
    Ui,Uie=run.jack_U(X,2,3); Up,Upe=run.jack_U(X,0,1)
    e,ee=run.jack_mean(X[4].mean(axis=1))
    print(f"T={T}: U_imp={Ui:.5f}({Uie:.5f}) U_plain={Up:.5f}({Upe:.5f}) e={e:.5f}({ee:.5f}) exact={ex.energy_per_spin_exact(T,N):.5f}  [{time.time()-t0:.0f}s]", flush=True)
print("total", time.time()-t00)
