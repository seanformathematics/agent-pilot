import sys, os, numpy as np
print(sys.version)
print(os.cpu_count())
print(np.__version__)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba", e)
