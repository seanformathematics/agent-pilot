import sys, os, shutil, platform
print("python", sys.version)
print("platform", platform.platform())
print("cpus", os.cpu_count())
for exe in ["gcc", "cc", "clang"]:
    print(exe, shutil.which(exe))
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba:", e)
import numpy as np
print("numpy", np.__version__)
