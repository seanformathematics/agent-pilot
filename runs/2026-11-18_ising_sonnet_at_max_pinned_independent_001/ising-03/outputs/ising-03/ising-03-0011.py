
import sys, pickle
sys.path.insert(0,'/work')
import numpy as np
import sw
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def load(label,k): return pickle.load(open(f'/work/prod{label}_k{k}.pkl','rb'))
print("(1) group-scatter calibration: 20 groups x 50 replicas (each 4000 sweeps)")
ratios=[]
for k,T in enumerate(Ts):
    A=load('A',k); B=load('B',k)
    blk=np.concatenate([A['blk'],B['blk']],axis=1)
    s2=blk[0].mean(axis=1); s4=blk[1].mean(axis=1)
    # shuffle replicas deterministically so groups mix hot/cold-start replicas
    rng=np.random.default_rng(4242+k); perm=rng.permutation(len(s2))
    s2=s2[perm]; s4=s4[perm]
    Ug=[]; eg=[]
    for g in range(20):
        sl=slice(g*50,(g+1)*50)
        U,e=sw.U_jack(s2[sl],s4[sl]); Ug.append(U); eg.append(e)
    Ug=np.array(Ug); eg=np.array(eg)
    r=Ug.std(ddof=1)/np.sqrt((eg**2).mean())
    ratios.append(r)
    print(f"T={T}: std over groups={Ug.std(ddof=1):.3e}  rms jackknife err per group={np.sqrt((eg**2).mean()):.3e}  ratio={r:.2f}")
ratios=np.array(ratios)
print("mean ratio over T: %.3f (expected 1, each ratio has ~16%% uncertainty => mean se ~0.045)"%ratios.mean())

print("\n(2) burn-in trajectories (ensemble mean of m^2, hot and cold starts), 200 discarded sweeps")
for k,T in enumerate(Ts):
    A=load('A',k); B=load('B',k)
    th=(A['traj_hot']+B['traj_hot'])/2; tc=(A['traj_cold']+B['traj_cold'])/2
    ref=(th[100:].mean()+tc[100:].mean())/2
    # first sweep index at which |hot-ref| and |cold-ref| both below 3*stat error (rough: use std of late values)
    sd=max(th[100:].std(),tc[100:].std())
    ok=np.where((np.abs(th-ref)<3*sd)&(np.abs(tc-ref)<3*sd))[0]
    # find first t after which all are ok
    bad=np.where(~((np.abs(th-ref)<3*sd)&(np.abs(tc-ref)<3*sd)))[0]
    t_conv=bad.max()+1 if len(bad) else 0
    print(f"T={T}: m2(eq)={ref:.5f}  mean m2 hot(t=100..200)={th[100:].mean():.5f} cold={tc[100:].mean():.5f}; last sweep outside 3*sd band: t={t_conv}; hot[0,1,2,5,10,20]={th[[0,1,2,5,10,20]].round(3)} cold={tc[[0,1,2,5,10,20]].round(3)}")
