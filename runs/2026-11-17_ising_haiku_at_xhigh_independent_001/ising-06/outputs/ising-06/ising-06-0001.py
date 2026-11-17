import sys, os, time
print(sys.version)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba", e)
import numpy as np
print("numpy", np.__version__)
print("cpus", os.cpu_count())
os.makedirs("/work", exist_ok=True)
print(os.listdir("/work"))
