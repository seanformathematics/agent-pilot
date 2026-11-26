import sys, time, numpy as np
print(sys.version, np.__version__)
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
t=time.time(); s=0
for i in range(10**7): s+=i
print("loop 1e7 s", time.time()-t)
