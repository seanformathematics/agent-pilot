
import sys, time, numpy as np
sys.path.insert(0,'/work')
from ising import run
res={}
for T,seed in [(1.9,101),(2.0,102),(2.1,103)]:
    t=time.time()
    R=400; nb=10; per=1500
    s=None
    B2=[];B4=[];BA=[]
    # burn-in
    s,_,_,_ = run(T,R,3000,0,seed)  # returns state after burn-in (nmeas=0 -> arrays of zeros)
    rng_seed=seed
    for b in range(nb):
        s,m2,m4,am = run(T,R,0,per,seed*1000+b,state=s) if False else (None,None,None,None)
        break
    print("skip")
