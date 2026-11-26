import sys, numpy, scipy, os
print(sys.version, numpy.__version__, scipy.__version__, os.cpu_count())
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)