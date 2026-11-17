import sys, time; sys.path.insert(0,'/work')
import numpy as np
from ising import run, binder_jk
for i,T in enumerate([1.5,2.0,2.3,2.8]):
    t=time.time()
    m2,m4 = run(T,16,500,6000,500,seed=3000+i)
    U,err=binder_jk(m2,m4)
    # drift check: U over first vs second half of measurements
    h=m2.shape[0]//2
    U1,_=binder_jk(m2[:h],m4[:h]); U2,_=binder_jk(m2[h:],m4[h:])
    print(T, round(U,5), round(err,5), 'halves', round(U1,5), round(U2,5), round(time.time()-t,1), flush=True)
