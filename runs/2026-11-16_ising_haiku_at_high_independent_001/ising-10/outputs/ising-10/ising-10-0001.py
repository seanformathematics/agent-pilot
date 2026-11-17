import time, importlib
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
import numpy as np, os
print(np.__version__, os.cpu_count())
os.makedirs('/work', exist_ok=True)
