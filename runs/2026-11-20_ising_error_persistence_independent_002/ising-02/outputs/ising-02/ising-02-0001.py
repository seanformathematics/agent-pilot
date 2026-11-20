import sys, os
print(sys.version)
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
import numpy; print("numpy", numpy.__version__)
print(os.cpu_count())