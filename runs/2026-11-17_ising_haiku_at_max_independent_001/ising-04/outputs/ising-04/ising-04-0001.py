import sys, os, shutil
print("python", sys.version)
print("cpus", os.cpu_count())
try:
    import numpy
    print("numpy", numpy.__version__)
except Exception as e:
    print("numpy missing", e)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("numba missing", type(e).__name__)
for exe in ["gcc", "cc", "clang"]:
    print(exe, shutil.which(exe))
