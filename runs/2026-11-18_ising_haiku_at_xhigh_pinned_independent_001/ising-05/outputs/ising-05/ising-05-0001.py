import subprocess, shutil, sys, time
print(sys.version)
print("gcc:", shutil.which("gcc"), shutil.which("cc"), shutil.which("clang"))
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba", e)
import numpy as np
print("numpy", np.__version__)
t=time.time()
s=0
for i in range(10**7): s+=i
print("py loop 1e7:", time.time()-t)
