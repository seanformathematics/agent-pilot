
try:
    import numba, numpy
    print("numba", numba.__version__, numpy.__version__)
except Exception as e:
    print("no numba", e)
