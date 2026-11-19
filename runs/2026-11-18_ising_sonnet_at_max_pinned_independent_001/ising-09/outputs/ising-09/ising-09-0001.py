
import sys, shutil, subprocess, os, time
print(sys.version)
for exe in ["gcc","cc","clang","tcc","g++","rustc","go","julia"]:
    print(exe, shutil.which(exe))
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba:", e)
import numpy, scipy
print("numpy", numpy.__version__, "scipy", scipy.__version__)
print("cpu_count", os.cpu_count())
print(os.listdir("/work"))
t=time.time()
s=0
for i in range(10**7):
    s+=i
print("pure python 1e7 loop", time.time()-t)
try:
    import cffi
    print("cffi ok")
except Exception as e:
    print("no cffi", e)
try:
    import Cython
    print("cython ok")
except Exception as e:
    print("no cython", e)
