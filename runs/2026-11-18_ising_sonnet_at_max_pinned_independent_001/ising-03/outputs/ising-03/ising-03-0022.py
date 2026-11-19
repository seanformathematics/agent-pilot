
import sys, time, pickle
sys.path.insert(0,'/work')
import numpy as np
import sw, metro
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
k=Ts.index(2.8)
t0=time.time()
# (a) new cold start, fresh seed
rm=metro.metropolis_run(16,1000,2.8,seed=88001,n_equil=3000,n_meas=20000,start="cold",B=100)
pickle.dump(rm,open(f'/work/metro3_cold_k{k}.pkl','wb'))
s2=rm['blk'][0].mean(axis=1); s4=rm['blk'][1].mean(axis=1); U,e=sw.U_jack(s2,s4)
print(f"(a) T=2.8 Metropolis cold/new seed 88001: U={U:.5f}+-{e:.5f}  <m2>={s2.mean():.5f}  [{time.time()-t0:.0f}s]",flush=True)
# (b) exact re-run of old config
rb=metro.metropolis_run(16,1000,2.8,seed=33013,n_equil=2000,n_meas=20000,start="cold",B=100)
old=pickle.load(open(f'/work/metro_k{k}.pkl','rb'))
print("(b) re-run identical to stored old run? max |diff| of block data:", np.abs(rb['blk']-old['blk']).max(), f"[{time.time()-t0:.0f}s]")
s2=rb['blk'][0].mean(axis=1); s4=rb['blk'][1].mean(axis=1); U,e=sw.U_jack(s2,s4)
print(f"    re-run U={U:.5f}+-{e:.5f}")
