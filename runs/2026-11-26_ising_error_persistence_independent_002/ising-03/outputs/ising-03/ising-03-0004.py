import sys; sys.path.insert(0,'/work')
from ising import *
import json
out={}
t0=time.time()
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    seed=20000+int(round(T*100))
    S2,S4,cnt,mser=run_T(T,B=1000,nburn=2000,nmeas=4000,seed=seed)
    U,err=binder_jk(S2,S4,cnt)
    # consistency: first vs second half of measurement window
    h=cnt//2
    m=mser.astype(np.float64)
    def Uof(x):
        a2=np.mean(x**2); a4=np.mean(x**4); return 1-a4/(3*a2*a2)
    U1=Uof(m[:h]); U2=Uof(m[h:])
    out[T]=dict(U=float(U),err=float(err),seed=seed,U_half1=float(U1),U_half2=float(U2))
    print(f"T={T} U={U:.5f} err={err:.5f} halves={U1:.5f},{U2:.5f} t={time.time()-t0:.0f}",flush=True)
json.dump(out,open('/work/prod.json','w'),indent=1)
