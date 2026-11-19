
import sys, shutil, subprocess, os, time
print(sys.version)
import numpy, scipy
print("numpy", numpy.__version__, "scipy", scipy.__version__)
for mod in ["numba", "cython", "cffi", "pythran", "torch", "jax"]:
    try:
        __import__(mod)
        print(mod, "available")
    except Exception as e:
        print(mod, "NOT available", type(e).__name__)
for exe in ["gcc", "cc", "clang", "tcc", "g++", "make", "rustc", "go"]:
    print(exe, shutil.which(exe))
print("cpu count", os.cpu_count())
print(os.listdir("/work"))
t=time.time()
s=0
for i in range(3_000_000):
    s+=i
print("pure python 3M loop iterations: %.2fs"%(time.time()-t))
