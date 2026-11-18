import sys, os, shutil, subprocess, platform
print("python", sys.version)
print("platform", platform.platform())
for prog in ["gcc", "cc", "clang", "tcc", "g++"]:
    print(prog, shutil.which(prog))
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("numba unavailable:", repr(e))
import numpy, scipy
print("numpy", numpy.__version__, "scipy", scipy.__version__)
print("cpus", os.cpu_count())
print("cwd", os.getcwd(), os.listdir("."))
try:
    out = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
    print(out.stdout[:300], out.stderr[:300])
except Exception as e:
    print("gcc run failed", e)
