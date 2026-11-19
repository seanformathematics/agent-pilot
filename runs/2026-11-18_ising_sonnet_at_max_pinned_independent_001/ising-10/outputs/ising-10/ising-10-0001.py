
import sys, shutil, subprocess, os, platform, time
print(sys.version)
import numpy, scipy
print("numpy", numpy.__version__, "scipy", scipy.__version__)
for exe in ["gcc","cc","clang","tcc","g++","gfortran","rustc","go","cython","pypy3"]:
    print(exe, shutil.which(exe))
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba:", e)
try:
    import cffi
    print("cffi", cffi.__version__)
except Exception as e:
    print("no cffi:", e)
try:
    import Cython
    print("cython", Cython.__version__)
except Exception as e:
    print("no cython:", e)
print(platform.platform(), platform.machine())
print("cpu count", os.cpu_count())
try:
    print(open('/proc/cpuinfo').read()[:800])
except Exception as e:
    print(e)
print(subprocess.run("free -m; df -h /work /tmp /dev/shm", shell=True, capture_output=True, text=True).stdout)
# quick speed benchmark
t=time.time()
s=0
for i in range(10**7):
    s+=i
print("10^7 python loop iterations: %.2fs"%(time.time()-t))
