import sys, time, numpy as np
sys.path.insert(0,'/work')
from ising import run
# --- validation: exact 4x4 enumeration vs simulation
N=4; T=2.0
cfg = np.array([[1 if (k>>b)&1 else -1 for b in range(16)] for k in range(1<<16)],dtype=np.int64).reshape(-1,4,4)
E = -( (cfg*np.roll(cfg,1,1)).sum((1,2)) + (cfg*np.roll(cfg,1,2)).sum((1,2)) )
w = np.exp(-(E-E.min())/T); w/=w.sum()
m = cfg.reshape(-1,16).mean(1)
ex2=(w*m**2).sum(); ex4=(w*m**4).sum()
_,M2,M4,_ = run(T,2000,4,500,200,1,123)
print("4x4 check: exact U=%.5f  MC U=%.5f"%(1-ex4/(3*ex2**2), 1-M4.mean()/(3*M2.mean()**2)))
print("exact <m2>=%.5f MC=%.5f"%(ex2, M2.mean()))
# --- production
for T in [1.9,2.0,2.1]:
    t=time.time()
    s,M2,M4,Ma = run(T,1000,16,2000,2000,10,1000+int(T*100))
    np.savez('/work/ts_%.2f.npz'%T, M2=M2.astype(np.float32), M4=M4.astype(np.float32), Ma=Ma.astype(np.float32))
    U = 1-M4.mean()/(3*M2.mean()**2)
    print("T=%.2f U=%.5f  time=%.1f"%(T,U,time.time()-t), flush=True)
