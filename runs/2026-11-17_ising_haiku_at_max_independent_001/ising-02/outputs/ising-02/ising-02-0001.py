import sys, shutil
print("python", sys.version)
import numpy as np
print("numpy", np.__version__)
print("gcc", shutil.which("gcc"), "cc", shutil.which("cc"))
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba", e)
