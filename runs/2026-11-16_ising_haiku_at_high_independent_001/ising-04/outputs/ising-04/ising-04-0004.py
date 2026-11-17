import sys, time, json; sys.path.insert(0,'/work')
import numpy as np
from ising import run, binder_jk
Ts=[2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res={}
for i,T in enumerate(Ts):
    t=time.time()
    m2,m4 = run(T, 16, 2000, 1500, 2000, seed=2000+i)
    U,err = binder_jk(m2,m4)
    res[T]=dict(U=float(U),err=float(err),n=2000)
    print(T, round(U,5), round(err,5), round(time.time()-t,1), flush=True)
json.dump(res, open('/work/res_part2.json','w'))
