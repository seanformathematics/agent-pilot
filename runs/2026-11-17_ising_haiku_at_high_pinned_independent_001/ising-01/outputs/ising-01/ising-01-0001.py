import shutil, subprocess, sys, platform, os
print(sys.version, platform.platform(), os.cpu_count())
for c in ["gcc","cc","clang","tcc"]:
    print(c, shutil.which(c))
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
import numpy; print("numpy", numpy.__version__)
