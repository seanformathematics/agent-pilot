import sys, time, os
sys.path.insert(0,'/work')
import numpy as np, ising
os.makedirs('/work/res', exist_ok=True)
Ts = [2.2,2.3,2.4,2.5,2.6,2.7,2.8]
R=1024
for idx,T in enumerate(Ts):
    seed = 1007+idx
    t0=time.time()
    out = ising.run(T, R=R, n_equil=2000, n_meas=4000, seed=seed, start='random')
    U,err = ising.binder_jk(out['A'],out['B'],out['K'])
    np.savez(f'/work/res/T{T:.2f}_s{seed}.npz', A=out['A'], B=out['B'], Mabs=out['Mabs'], K=out['K'],
             ser_m2=out['ser_m2'], ser_m4=out['ser_m4'], ser_m1=out['ser_m1'])
    print(f"T={T} seed={seed} U={U:.5f} err={err:.5f} K={out['K']} tau_m2={ising.tau_int(out['ser_m2']):.2f} time={time.time()-t0:.1f}s", flush=True)
