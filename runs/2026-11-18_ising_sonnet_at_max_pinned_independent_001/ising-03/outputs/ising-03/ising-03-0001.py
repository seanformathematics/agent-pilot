
import sys, shutil, subprocess, os, time
print(sys.version)
for c in ["gcc","cc","clang","tcc","g++","gfortran","rustc","go"]:
    print(c, shutil.which(c))
try:
    import numpy, scipy
    print("numpy", numpy.__version__, "scipy", scipy.__version__)
except Exception as e:
    print("err", e)
for m in ["numba","cython","cffi","mpmath","pandas","matplotlib","sympy","torch","jax"]:
    try:
        __import__(m)
        print(m, "available")
    except Exception as e:
        print(m, "NOT available")
print("cpu count", os.cpu_count())
print(open("/proc/cpuinfo").read()[:600] if os.path.exists("/proc/cpuinfo") else "no cpuinfo")
print(subprocess.run(["df","-h","/work","/tmp"],capture_output=True,text=True).stdout)
t=time.time()
s=0
for i in range(10**7): s+=i
print("pure python loop 1e7:", time.time()-t)
