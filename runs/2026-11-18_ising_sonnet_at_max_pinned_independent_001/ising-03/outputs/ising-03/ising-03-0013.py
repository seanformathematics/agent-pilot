
import sys, time, pickle, os
sys.path.insert(0,'/work')
import numpy as np
import sw, metro
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def load(label,k): return pickle.load(open(f'/work/prod{label}_k{k}.pkl','rb'))
def sw_pooled(k):
    A=load('A',k); B=load('B',k)
    blk=np.concatenate([A['blk'],B['blk']],axis=1)
    return sw.U_jack(blk[0].mean(axis=1),blk[1].mean(axis=1))

def run_metro(idx_list, nmeas_of, tmax=520):
    t0=time.time()
    for k in idx_list:
        T=Ts[k]; fn=f'/work/metro_k{k}.pkl'
        if os.path.exists(fn): continue
        if time.time()-t0>tmax: print("time guard before T",T); break
        r=metro.metropolis_run(16,1000,T,seed=33000+k,n_equil=2000,n_meas=nmeas_of(T),start="cold",B=100)
        pickle.dump(r,open(fn,'wb'))
        s2=r['blk'][0].mean(axis=1); s4=r['blk'][1].mean(axis=1)
        U,e=sw.U_jack(s2,s4)
        Us,es=sw_pooled(k)
        nb=r['blk'].shape[2]
        Ue,ee=sw.U_jack(r['blk'][0][:,:nb//2].mean(axis=1),r['blk'][1][:,:nb//2].mean(axis=1))
        Ul,el=sw.U_jack(r['blk'][0][:,nb//2:].mean(axis=1),r['blk'][1][:,nb//2:].mean(axis=1))
        print(f"T={T}: Metropolis U={U:.6f} +- {e:.6f} | SW pooled U={Us:.6f} +- {es:.6f} | z={(U-Us)/np.hypot(e,es):+.2f} | Metro early-late z={(Ue-Ul)/np.hypot(ee,el):+.2f}  ({time.time()-t0:.0f}s)",flush=True)

nm=lambda T: 10000 if T<=2.0 else 20000
run_metro(range(0,8), nm)
