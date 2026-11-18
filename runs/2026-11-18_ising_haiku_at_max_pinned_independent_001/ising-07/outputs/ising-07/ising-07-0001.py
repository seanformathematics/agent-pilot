import sys, os, shutil, subprocess, platform
print("python", sys.version)
print("platform", platform.platform())
print("cpu_count", os.cpu_count())
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
try:
    import scipy
    print("scipy", scipy.__version__)
except Exception as e:
    print("scipy missing", e)
for c in ["gcc", "cc", "clang", "tcc"]:
    print(c, shutil.which(c))
print("cwd", os.getcwd())
print("work exists", os.path.isdir("/work"), os.access("/work", os.W_OK) if os.path.isdir("/work") else None)
print(os.listdir("/work") if os.path.isdir("/work") else "no /work")
