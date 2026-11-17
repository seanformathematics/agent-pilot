import sys, shutil, os
print(sys.version)
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
print(os.cpu_count())
