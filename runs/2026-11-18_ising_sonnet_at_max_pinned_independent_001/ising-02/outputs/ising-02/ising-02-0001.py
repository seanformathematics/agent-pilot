
import sys, shutil, subprocess, os, time
print(sys.version)
for tool in ["gcc","cc","clang","tcc","g++","cython","rustc","go"]:
    print(tool, shutil.which(tool))
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba:", e)
try:
    import numpy, scipy
    print("numpy", numpy.__version__, "scipy", scipy.__version__)
except Exception as e:
    print(e)
try:
    import cffi
    print("cffi", cffi.__version__)
except Exception as e:
    print("no cffi:", e)
print(os.cpu_count())
print(subprocess.run("cat /proc/cpuinfo | grep 'model name' | head -2; free -m; df -h /work /tmp", shell=True, capture_output=True, text=True).stdout)
# quick speed test of pure python loop
t=time.time()
s=0
for i in range(5_000_000):
    s+=i&3
print("5M loop iters:", time.time()-t)
