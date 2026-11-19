
import sys, time, pickle, os
sys.path.insert(0,'/work')
import numpy as np
import sw, metro
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def load(label,k): return pickle.load(open(f'/work/prod{label}_k{k}.pkl','rb'))
t0=time.time()
zs=[]
for T in (2.1,2.2,2.3,2.4,2.6,2.7):
    k=Ts.index(T)
    start="cold" if T<=2.3 else "hot"
    rm=metro.metropolis_run(16,1000,T,seed=99000+k,n_equil=2000,n_meas=16000,start=start,B=100)
    pickle.dump(rm,open(f'/work/metro4_k{k}.pkl','wb'))
    s2=rm['blk'][0].mean(axis=1); s4=rm['blk'][1].mean(axis=1); U,e=sw.U_jack(s2,s4)
    A=load('A',k); B=load('B',k); bb=np.concatenate([A['blk'],B['blk']],axis=1)
    Up,ep=sw.U_jack(bb[0].mean(axis=1),bb[1].mean(axis=1)); Ui,ei=sw.U_jack(bb[2].mean(axis=1),bb[3].mean(axis=1))
    zp=(U-Up)/np.hypot(e,ep); zi=(U-Ui)/np.hypot(e,ei); zs.append(zi)
    print(f"T={T} ({start}) new Metro U={U:.5f}+-{e:.5f} | SW plain {Up:.5f}+-{ep:.5f} z={zp:+.2f} | SW impr {Ui:.5f}+-{ei:.5f} z={zi:+.2f}   [{time.time()-t0:.0f}s]",flush=True)
zs=np.array(zs); print("chi2 of z (vs SW improved) = %.1f for %d"%((zs**2).sum(),len(zs)))
