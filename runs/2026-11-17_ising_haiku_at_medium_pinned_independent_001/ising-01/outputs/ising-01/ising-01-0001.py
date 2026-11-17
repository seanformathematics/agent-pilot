import shutil, sys, platform, os
print(sys.version)
print(platform.platform())
for p in ["gcc","cc","clang","tcc"]:
    print(p, shutil.which(p))
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba", e)
import numpy
print("numpy", numpy.__version__)
print(os.cpu_count())
