import sys
print(sys.version)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("numba missing", e)
