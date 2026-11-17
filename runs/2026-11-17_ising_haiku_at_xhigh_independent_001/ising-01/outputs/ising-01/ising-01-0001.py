import shutil, subprocess, sys, os
print(sys.version)
for c in ["gcc","cc","clang","tcc"]:
    print(c, shutil.which(c))
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
import numpy; print("numpy", numpy.__version__)
print(os.cpu_count())
print(subprocess.run(["bash","-lc","cat /proc/cpuinfo | grep 'model name' | head -1; free -m"],capture_output=True,text=True).stdout)