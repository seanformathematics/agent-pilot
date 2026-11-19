
import re, sys, time, pickle
# --- patch metro.py: optional energy measurement
src=open('/work/metro.py').read()
src=src.replace('def metropolis_run(N, R, T, seed, n_equil, n_meas, start="hot", B=100):','def metropolis_run(N, R, T, seed, n_equil, n_meas, start="hot", B=100, e_every=0):')
src=src.replace("    blk = np.zeros((2, R, nb))\n","    blk = np.zeros((2, R, nb))\n    eacc_tot = np.zeros(R); ecount = 0\n")
src=src.replace("            a2 += m2; a4 += m2 * m2\n","            a2 += m2; a4 += m2 * m2\n            if e_every and (t % e_every) == 0:\n                s16 = s.astype(np.int16)\n                eacc_tot += -(s16 * np.roll(s16, -1, axis=2) + s16 * np.roll(s16, -1, axis=1)).sum(axis=(1, 2)) / float(N * N); ecount += 1\n")
src=src.replace("    return dict(blk=blk, N=N, R=R, T=T, seed=seed, n_equil=n_equil, n_meas=nb * B, B=B, start=start)","    return dict(blk=blk, N=N, R=R, T=T, seed=seed, n_equil=n_equil, n_meas=nb * B, B=B, start=start, e_rep=(eacc_tot / max(ecount, 1)))")
open('/work/metro.py','w').write(src)

sys.path.insert(0,'/work')
import numpy as np
import sw, metro, exact
def load(label,k): return pickle.load(open(f'/work/prod{label}_k{k}.pkl','rb'))
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
t0=time.time()
for T,seedM,seedC in ((2.8,55001,56001),(2.5,55002,56002)):
    k=Ts.index(T)
    # new Metropolis from HOT start, with energies
    rm=metro.metropolis_run(16,1000,T,seed=seedM,n_equil=3000,n_meas=20000,start="hot",B=100,e_every=5)
    pickle.dump(rm,open(f'/work/metro2_k{k}.pkl','wb'))
    s2=rm['blk'][0].mean(axis=1); s4=rm['blk'][1].mean(axis=1)
    Um,em=sw.U_jack(s2,s4)
    ee=exact.energy_per_site_exact(T,16)
    er=rm['e_rep']; ze=(er.mean()-ee)/(er.std(ddof=1)/np.sqrt(len(er)))
    print(f"T={T} Metropolis(hot start, new seed): U={Um:.5f}+-{em:.5f}; energy {er.mean():.6f}+-{er.std(ddof=1)/np.sqrt(len(er)):.6f} vs exact {ee:.6f} (z={ze:+.2f})   [{time.time()-t0:.0f}s]",flush=True)
    # new SW set C
    rc=sw.run_T(16,500,T,seed=seedC,n_equil=200,n_meas=8000,B=100,e_every=5)
    pickle.dump(rc,open(f'/work/prodC_k{k}.pkl','wb'))
    blk=rc['blk']
    Up,ep=sw.U_jack(blk[0].mean(axis=1),blk[1].mean(axis=1)); Ui,ei=sw.U_jack(blk[2].mean(axis=1),blk[3].mean(axis=1))
    er=rc['eblk'].mean(axis=1); ze=(er.mean()-ee)/(er.std(ddof=1)/np.sqrt(len(er)))
    print(f"T={T} SW set C (new seed, 4e6 samples): plain U={Up:.5f}+-{ep:.5f}; improved U={Ui:.5f}+-{ei:.5f}; energy z={ze:+.2f}   [{time.time()-t0:.0f}s]",flush=True)
    # compare with old
    A=load('A',k); B=load('B',k)
    bb=np.concatenate([A['blk'],B['blk']],axis=1)
    Uai,eai=sw.U_jack(bb[2].mean(axis=1),bb[3].mean(axis=1))
    M1=pickle.load(open(f'/work/metro_k{k}.pkl','rb')); Um1,em1=sw.U_jack(M1['blk'][0].mean(axis=1),M1['blk'][1].mean(axis=1))
    print(f"      old SW(A+B) improved {Uai:.5f}+-{eai:.5f}; old Metro {Um1:.5f}+-{em1:.5f}")
    print(f"      z(newMetro - newSWimpr) = {(Um-Ui)/np.hypot(em,ei):+.2f};  z(newMetro - oldMetro) = {(Um-Um1)/np.hypot(em,em1):+.2f}; z(newSWimpr - oldSWimpr) = {(Ui-Uai)/np.hypot(ei,eai):+.2f}",flush=True)
