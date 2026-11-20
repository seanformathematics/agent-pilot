import sys, time, numpy as np
sys.path.insert(0,'/work')
from wolff import run
for T in [1.9,2.3]:
    t0=time.time(); m2,m4=run(16,T,100000,7,2000)
    print(T, time.time()-t0, 1-m4.mean()/(3*m2.mean()**2), flush=True)
