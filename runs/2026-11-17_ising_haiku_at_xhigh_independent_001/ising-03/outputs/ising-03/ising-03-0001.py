import sys, time, os
print(sys.version)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba", e)
import numpy as np
print("numpy", np.__version__)
print(os.cpu_count())
t=time.time()
s=0
for i in range(10**7):
    s+=i
print("loop 1e7", time.time()-t)
