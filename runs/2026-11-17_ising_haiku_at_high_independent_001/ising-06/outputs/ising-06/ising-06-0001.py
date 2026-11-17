import time, sys
print(sys.version)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba", e)
import numpy as np
print(np.__version__)
t=time.time(); s=0
for i in range(10**7): s+=i
print("loop 1e7", time.time()-t)
