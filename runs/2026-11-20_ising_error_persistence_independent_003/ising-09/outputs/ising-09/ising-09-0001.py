import os
print(os.cpu_count())
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba", e)
