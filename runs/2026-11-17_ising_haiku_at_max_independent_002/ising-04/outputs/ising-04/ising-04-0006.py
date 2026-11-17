import sys, time, json
sys.path.insert(0, "/work")
import numpy as np
import wolff

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
res = []
t_all = time.time()
for idx, T in enumerate(Ts):
    t0 = time.time()
    M, _ = wolff.run_chain(16, T, seed=1000+idx, n_equil=5000, n_meas=200000, init=0)
    dt = time.time()-t0
    U, err, K = wolff.binder_jk(M, 2000)
    tau2 = wolff.tau_int((M.astype(float))**2)
    tau4 = wolff.tau_int((M.astype(float))**4)
    res.append(dict(T=T, U=U, err=err, tau2=tau2, tau4=tau4, time=dt))
    print("T", T, "U", round(U, 5), "err", round(err, 5), "tau_m2", round(tau2, 2), "tau_m4", round(tau4, 2), "time", round(dt, 1), "meanm2", round(float(np.mean((M.astype(float)/256)**2)), 5))
    if time.time()-t_all > 420:
        print("stopping early for time")
        break
with open("/work/pilot.json", "w") as f:
    json.dump(res, f)
