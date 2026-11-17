import sys, os, shutil, subprocess, platform
print("python", sys.version)
print("platform", platform.platform())
print("cpu count", os.cpu_count())
try:
    import numpy as np
    print("numpy", np.__version__)
except Exception as e:
    print("numpy missing", e)
try:
    import scipy
    print("scipy", scipy.__version__)
except Exception as e:
    print("scipy missing", e)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("numba missing", e)
for c in ["gcc", "cc", "clang", "tcc"]:
    print(c, shutil.which(c))
try:
    out = subprocess.run(["gcc", "--version"], capture_output=True, text=True, timeout=20)
    print(out.stdout.splitlines()[0] if out.stdout else out.stderr)
except Exception as e:
    print("gcc run failed", e)
print(open("/proc/meminfo").read().splitlines()[0])
print("cwd", os.getcwd(), os.listdir("/work") if os.path.exists("/work") else "no /work")
