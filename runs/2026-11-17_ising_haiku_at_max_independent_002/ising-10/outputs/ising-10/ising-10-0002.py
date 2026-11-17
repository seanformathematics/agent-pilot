import sys, subprocess, os, platform
try:
    import numpy as np
    print("numpy", np.__version__)
except Exception as e:
    print("numpy missing", e)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("numba missing", e)
try:
    import scipy
    print("scipy", scipy.__version__)
except Exception as e:
    print("scipy missing", e)
for cc in ["gcc", "cc", "clang", "tcc"]:
    r = subprocess.run(["which", cc], capture_output=True, text=True)
    print(cc, "->", r.stdout.strip() or "not found")
print("platform", platform.platform())
try:
    print(open("/proc/cpuinfo").read().split("\n\n")[0][:500])
except Exception as e:
    print(e)
print("affinity", len(os.sched_getaffinity(0)))
