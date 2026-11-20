import subprocess,sys
print(sys.version)
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
print(subprocess.run("which gcc cc clang; nproc; python3 -c 'import numpy;print(numpy.__version__)'",shell=True,capture_output=True,text=True).stdout)