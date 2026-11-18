import sys, time, importlib
print(sys.version)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba", e)
import numpy as np, scipy
print("numpy", np.__version__, "scipy", scipy.__version__)
t=time.time()
s=0
for i in range(10**7):
    s+=i
print("loop 1e7 time", time.time()-t)
import os
print(os.cpu_count())
