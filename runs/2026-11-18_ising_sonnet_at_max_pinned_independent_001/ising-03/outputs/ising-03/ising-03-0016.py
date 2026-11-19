
import sys, time, pickle
sys.path.insert(0,'/work')
import numpy as np
import sw, metro, exactM
t0=time.time()
ex={}
for T in (2.8,2.5,2.3):
    ex[T]=exactM.moments_from_ZM(8,1/T)
    print(f"exact N=8 T={T}: m2={ex[T][0]:.8f} m4={ex[T][1]:.8f} U={ex[T][2]:.8f}  ({time.time()-t0:.0f}s)",flush=True)
res={}
for i,T in enumerate((2.8,2.5,2.3)):
    m2e,m4e,Ue=ex[T]
    # SW
    t1=time.time()
    r=sw.run_T(8,1000,T,seed=44000+i,n_equil=200,n_meas=10000,B=100,e_every=100)
    blk=r['blk']
    Up,ep=sw.U_jack(blk[0].mean(axis=1),blk[1].mean(axis=1)); Ui,ei=sw.U_jack(blk[2].mean(axis=1),blk[3].mean(axis=1))
    print(f"N=8 T={T} SW   plain U={Up:.5f}+-{ep:.5f} (z={(Up-Ue)/ep:+.2f}) | improved U={Ui:.5f}+-{ei:.5f} (z={(Ui-Ue)/ei:+.2f}) | m2: {blk[0].mean():.5f} (exact {m2e:.5f}) m4: {blk[1].mean():.5f} (exact {m4e:.5f})  [{time.time()-t1:.0f}s]",flush=True)
    t1=time.time()
    rm=metro.metropolis_run(8,1000,T,seed=45000+i,n_equil=2000,n_meas=20000,start="cold",B=100)
    s2=rm['blk'][0].mean(axis=1); s4=rm['blk'][1].mean(axis=1)
    Um,em=sw.U_jack(s2,s4)
    print(f"N=8 T={T} Metro      U={Um:.5f}+-{em:.5f} (z={(Um-Ue)/em:+.2f}) | m2: {s2.mean():.5f} m4: {s4.mean():.5f}  [{time.time()-t1:.0f}s]",flush=True)
