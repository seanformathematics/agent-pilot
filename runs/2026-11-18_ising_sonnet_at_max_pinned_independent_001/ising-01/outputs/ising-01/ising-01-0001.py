
import sys, shutil, os, platform, time
print(sys.version)
print(platform.platform())
for mod in ["numpy","scipy","numba","cython","cffi","pythran"]:
    try:
        m = __import__(mod)
        print(mod, getattr(m,"__version__","?"))
    except Exception as e:
        print(mod, "NOT AVAILABLE", type(e).__name__, e)
for c in ["gcc","cc","clang","tcc","g++","gfortran","rustc","go"]:
    print(c, shutil.which(c))
print("cpu_count", os.cpu_count())
try:
    print(open("/proc/cpuinfo").read()[:800])
except Exception as e:
    print(e)
try:
    print(open("/proc/meminfo").read()[:200])
except Exception as e:
    print(e)
print(os.listdir("/work"))
