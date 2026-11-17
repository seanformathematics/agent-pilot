import subprocess, sys, os
print(sys.version)
print(os.cpu_count())
for cmd in ["which gcc cc clang tcc", "gcc --version | head -1", "python -c 'import numba; print(numba.__version__)'", "python -c 'import numpy; print(numpy.__version__)'", "nproc", "free -m"]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(cmd, "->", r.stdout.strip(), r.stderr.strip()[:200])
