import sys, os, shutil, subprocess, platform
print("python", sys.version)
print("platform", platform.platform())
print("cpus", os.cpu_count())
try:
    import numpy as np
    print("numpy", np.__version__)
except Exception as e:
    print("numpy missing", e)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("numba missing:", e)
try:
    import scipy
    print("scipy", scipy.__version__)
except Exception as e:
    print("scipy missing", e)
for c in ["gcc", "cc", "clang", "tcc"]:
    print(c, shutil.which(c))
print(subprocess.run(["bash","-lc","cat /proc/cpuinfo | grep 'model name' | head -2; free -m; ls -la /work 2>/dev/null; pwd"],capture_output=True,text=True).stdout)
