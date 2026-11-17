import sys, os, shutil, platform
print("python", sys.version)
print("cpu count", os.cpu_count(), "affinity", len(os.sched_getaffinity(0)))
try:
    import numpy as np
    print("numpy", np.__version__)
except Exception as e:
    print("numpy missing", e)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("numba missing:", e)
for c in ["gcc", "cc", "clang", "tcc"]:
    print(c, shutil.which(c))
print(open("/proc/meminfo").read().splitlines()[:2])
