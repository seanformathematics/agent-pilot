import time, importlib
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
import numpy as np; print(np.__version__)
t=time.time(); a=np.random.default_rng(1).random((1000,16,16)); s=a.sum(); print(time.time()-t)