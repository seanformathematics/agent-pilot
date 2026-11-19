
import sys, time, pickle
sys.path.insert(0,'/work')
import numpy as np
import sw, metro
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def load(label,k): return pickle.load(open(f'/work/prod{label}_k{k}.pkl','rb'))
# (1) improved-estimator A vs B z-scores
zi=[]; zp=[]
for k,T in enumerate(Ts):
    A=load('A',k); B=load('B',k)
    a=sw.U_jack(A['blk'][2].mean(axis=1),A['blk'][3].mean(axis=1)); b=sw.U_jack(B['blk'][2].mean(axis=1),B['blk'][3].mean(axis=1))
    zi.append((a[0]-b[0])/np.hypot(a[1],b[1]))
    a=sw.U_jack(A['blk'][0].mean(axis=1),A['blk'][1].mean(axis=1)); b=sw.U_jack(B['blk'][0].mean(axis=1),B['blk'][1].mean(axis=1))
    zp.append((a[0]-b[0])/np.hypot(a[1],b[1]))
zi=np.array(zi); zp=np.array(zp)
print("A-vs-B z (improved):",zi.round(2)," chi2=%.1f/14"%(zi**2).sum())
print("A-vs-B z (plain)   :",zp.round(2)," chi2=%.1f/14"%(zp**2).sum(),flush=True)

# (2) Metropolis replicates at T=2.5
t0=time.time()
k=Ts.index(2.5)
for label,start,seed in (("cold/new seed","cold",77001),("hot/new seed","hot",77002)):
    rm=metro.metropolis_run(16,1000,2.5,seed=seed,n_equil=3000,n_meas=20000,start=start,B=100)
    pickle.dump(rm,open(f'/work/metro3_{start}_k{k}.pkl','wb'))
    s2=rm['blk'][0].mean(axis=1); s4=rm['blk'][1].mean(axis=1)
    U,e=sw.U_jack(s2,s4)
    print(f"T=2.5 Metropolis {label}: U={U:.5f}+-{e:.5f}  <m2>={s2.mean():.5f}+-{s2.std(ddof=1)/np.sqrt(len(s2)):.5f} <m4>={s4.mean():.6f}  [{time.time()-t0:.0f}s]",flush=True)
