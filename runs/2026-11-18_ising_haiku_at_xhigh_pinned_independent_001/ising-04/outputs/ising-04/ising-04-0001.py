import subprocess, sys, os
print(sys.version)
for cmd in ["gcc --version", "cc --version", "clang --version", "which gcc cc tcc"]:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        print(cmd, "->", r.returncode, r.stdout[:200], r.stderr[:200])
    except Exception as e:
        print(cmd, e)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba", e)
import numpy; print("numpy", numpy.__version__)
print(os.cpu_count())
