import subprocess, sys, os, platform
print(sys.version)
print(platform.platform())
for cmd in ["which gcc cc clang tcc", "gcc --version", "nproc", "cat /proc/cpuinfo | grep 'model name' | head -2", "free -m"]:
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20)
        print(">>", cmd, "\n", out.stdout, out.stderr)
    except Exception as e:
        print("ERR", cmd, e)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba:", e)
import numpy, scipy
print("numpy", numpy.__version__, "scipy", scipy.__version__)
print(os.getcwd(), os.listdir('.'))
