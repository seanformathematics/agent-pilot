import subprocess, sys
print(sys.version)
print(subprocess.run("which gcc cc clang; gcc --version | head -1; nproc", shell=True, capture_output=True, text=True).stdout)
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
import numpy; print("numpy", numpy.__version__)
