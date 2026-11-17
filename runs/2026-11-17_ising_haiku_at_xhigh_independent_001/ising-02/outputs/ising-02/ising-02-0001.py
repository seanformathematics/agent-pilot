import shutil, subprocess, sys
print(sys.version)
for c in ["gcc","cc","clang","tcc"]:
    print(c, shutil.which(c))
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba", e)
import numpy as np
print("numpy", np.__version__)
import os
print(os.cpu_count())
