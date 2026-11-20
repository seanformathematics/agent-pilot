import sys, json, time, numpy as np
sys.path.insert(0,'/work')
from ising import run, binder, jack_err
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
res={}
for idx,T in enumerate(Ts):
    t=time.time()
    seed=1000+idx
    m2b,m4b,_=run(T, seed=seed, R=1000, n_eq=1500, n_meas=6000, block=20)
    U,e=jack_err(m2b,m4b)
    # drift check: first vs second half
    h=m2b.shape[0]//2
    U1=binder(m2b[:h],m4b[:h]); U2=binder(m2b[h:],m4b[h:])
    res[T]=dict(seed=seed,U=float(U),err=float(e),U_half1=float(U1),U_half2=float(U2),
                m2=float(m2b.mean()), n_meas=6000, R=1000)
    print(T, "U=%.5f err=%.5f halves %.5f %.5f  t=%.1f"%(U,e,U1,U2,time.time()-t), flush=True)
json.dump(res,open('/work/results.json','w'),indent=1)
