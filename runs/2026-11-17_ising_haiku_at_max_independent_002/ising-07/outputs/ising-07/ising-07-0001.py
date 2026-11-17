import sys, platform, os, shutil, subprocess
print(sys.version)
print(platform.platform())
import numpy, scipy
print("numpy", numpy.__version__, "scipy", scipy.__version__)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba:", repr(e))
for c in ["gcc", "cc", "clang", "tcc", "g++"]:
    print(c, shutil.which(c))
print("cpu_count", os.cpu_count())
os.makedirs("/work", exist_ok=True)
print("work:", os.listdir("/work"))
try:
    out = subprocess.run(["gcc", "--version"], capture_output=True, text=True, timeout=20)
    print(out.stdout[:300])
except Exception as e:
    print("gcc run failed:", repr(e))
