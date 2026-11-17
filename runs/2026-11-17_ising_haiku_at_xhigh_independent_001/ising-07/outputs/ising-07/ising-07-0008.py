import sys, json; sys.path.insert(0,'/work')
import numpy as np, ising, time
Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res = []
t0=time.time()
for i,T in enumerate(Ts):
    ms = ising.run(T, R=400, S=2000, burn=300, seed=1000+i, N=16)
    U, Ue, Uj = ising.jack_U(ms)
    # consistency: halves of replicas
    U1,e1,_ = ising.jack_U(ms[:, :200]); U2,e2,_ = ising.jack_U(ms[:, 200:])
    # time-block check: first vs second half of sweeps
    Ua = ising.binder_from(ms[:1000]); Ub = ising.binder_from(ms[1000:])
    r = dict(T=T, U=float(U), U_err=float(Ue), n=int(ms.size), meanabsm=float(np.abs(ms).mean()),
             U_halfR=[float(U1),float(U2)], U_halfS=[float(Ua),float(Ub)])
    res.append(r)
    print(r, "t=%.1f"%(time.time()-t0), flush=True)
    with open('/work/results_N16.json','w') as f:
        json.dump(res,f,indent=1)
