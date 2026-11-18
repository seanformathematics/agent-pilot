import sys, os, shutil, subprocess, time, platform
print("python", sys.version)
print("platform", platform.platform())
try:
    import numpy as np; print("numpy", np.__version__)
except Exception as e: print("numpy missing", e)
try:
    import scipy; print("scipy", scipy.__version__)
except Exception as e: print("scipy missing", e)
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("numba not available:", e)
for c in ["gcc","cc","clang","tcc","g++"]:
    print(c, shutil.which(c))
print("cpu count", os.cpu_count())
try:
    print(open("/proc/cpuinfo").read().split("\n")[4])
except Exception as e: print(e)
os.makedirs("/work", exist_ok=True)
print("work writable", os.access("/work", os.W_OK))
t=time.time(); s=0
for i in range(10_000_000): s+=i
print("10M python loop s:", round(time.time()-t,3))
