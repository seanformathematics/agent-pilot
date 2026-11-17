import subprocess,sys
print(sys.version)
for c in ["gcc --version","cc --version","clang --version"]:
    try:
        print(c, subprocess.run(c,shell=True,capture_output=True,text=True).stdout[:200])
    except Exception as e: print(e)
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
import os; print(os.cpu_count())