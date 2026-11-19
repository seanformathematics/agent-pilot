import shutil, subprocess, sys
print(sys.version)
print("gcc:", shutil.which("gcc"), shutil.which("cc"), shutil.which("clang"))
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
import numpy; print("numpy", numpy.__version__)
import os; print(os.cpu_count())