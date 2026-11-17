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
a=np.random.default_rng(1).random((1000,16,16))
for _ in range(100):
    b=a*a+a
print("time per op on 1000x16x16:", (time.time()-t)/100)
