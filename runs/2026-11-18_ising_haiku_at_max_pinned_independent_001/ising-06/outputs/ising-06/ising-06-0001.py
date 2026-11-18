import sys, shutil, subprocess, os, platform
print("python", sys.version)
print("platform", platform.platform())
for c in ["gcc", "cc", "clang", "tcc"]:
    print(c, shutil.which(c))
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba:", e)
import numpy, scipy
print("numpy", numpy.__version__, "scipy", scipy.__version__)
print("cpu count", os.cpu_count())
try:
    print(open("/proc/cpuinfo").read().split("\n\n")[0][:600])
except Exception as e:
    print(e)
os.makedirs("/work", exist_ok=True)
print(os.listdir("/work"))
