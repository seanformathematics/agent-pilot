import sys, time, json; sys.path.insert(0,'/work')
import numpy as np
from ising import run, binder_jk
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1]
res={}
for i,T in enumerate(Ts):
    t=time.time()
    m2,m4 = run(T, 16, 2000, 1500, 2000, seed=1000+i)
    U,err = binder_jk(m2,m4)
    res[T]=dict(U=float(U),err=float(err),n=2000)
    print(T, round(U,5), round(err,5), round(time.time()-t,1), flush=True)
json.dump(res, open('/work/res_part1.json','w'))
