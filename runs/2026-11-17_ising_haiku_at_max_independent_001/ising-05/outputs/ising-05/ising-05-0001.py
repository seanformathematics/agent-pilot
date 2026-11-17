import sys, platform, shutil, subprocess, time, os
print("python", sys.version)
print("platform", platform.platform())
print("cpu count", os.cpu_count())
try:
    import numpy as np
    print("numpy", np.__version__)
except Exception as e:
    print("numpy missing", e)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("numba not available:", e)
try:
    import scipy
    print("scipy", scipy.__version__)
except Exception as e:
    print("scipy missing", e)
for c in ["gcc", "cc", "clang", "tcc"]:
    print(c, shutil.which(c))
# rough speed test: pure python loop
t0 = time.time()
s = 0
for i in range(10**6):
    s += i
print("1e6 python loop iterations took", round(time.time()-t0, 3), "s")
os.makedirs("/work", exist_ok=True)
print("work dir exists:", os.path.isdir("/work"), os.listdir("/work") if os.path.isdir("/work") else None)
