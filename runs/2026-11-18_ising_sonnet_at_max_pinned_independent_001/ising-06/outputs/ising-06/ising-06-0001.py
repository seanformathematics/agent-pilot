
import sys, os, shutil, subprocess, time
print(sys.version)
import numpy, scipy
print("numpy", numpy.__version__, "scipy", scipy.__version__)
for mod in ["numba", "cython", "cffi", "pythran", "torch", "jax"]:
    try:
        __import__(mod)
        print(mod, "available")
    except Exception as e:
        print(mod, "NOT available:", type(e).__name__)
for exe in ["gcc", "cc", "clang", "tcc", "g++", "make"]:
    print(exe, shutil.which(exe))
print("cpu count", os.cpu_count())
print(os.listdir("/work"))
t=time.time()
x=0
for i in range(3_000_000):
    x+=i
print("py loop 3M iters:", time.time()-t)
try:
    print(open("/proc/cpuinfo").read()[:600])
except Exception as e:
    print(e)
try:
    print(open("/proc/meminfo").read()[:200])
except Exception as e:
    print(e)
