
import sys, os, shutil, subprocess, platform
print("python", sys.version)
print("platform", platform.platform())
print("cpu count", os.cpu_count())
for c in ["gcc", "cc", "clang", "tcc", "g++"]:
    print(c, shutil.which(c))
try:
    import numpy as np
    print("numpy", np.__version__)
except Exception as e:
    print("numpy err", e)
try:
    import scipy
    print("scipy", scipy.__version__)
except Exception as e:
    print("scipy err", e)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("numba not available:", e)
print("cwd", os.getcwd())
print(os.listdir("/work") if os.path.exists("/work") else "no /work")
