import numpy as np
print(np.__version__)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba", e)
