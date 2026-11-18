import sys, os, numpy as np, scipy
print(sys.version)
print(np.__version__, scipy.__version__)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba", e)
print(os.cpu_count(), os.getcwd(), os.listdir('.'))
