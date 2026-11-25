
import numpy as np, json
exec(open('/work/sim.py').read())
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
R=512
res={}
for k,T in enumerate(Ts):
    seed=1001+k
    M2,M4=run(T,R,10000,2000,seed,block=10,start='random')
    # pooled estimator
    a2=M2.mean(0); a4=M4.mean(0)   # per-replica time averages
    def U_of(idx):
        m2=a2[idx].mean(); m4=a4[idx].mean()
        return 1-m4/(3*m2*m2)
    allidx=np.arange(R)
    U=U_of(allidx)
    jk=np.array([U_of(np.delete(allidx,r)) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    res[T]=dict(U=float(U),err=float(err),seed=seed)
    print(T, U, err, flush=True)
json.dump(res,open('/work/res1.json','w'),indent=1)
