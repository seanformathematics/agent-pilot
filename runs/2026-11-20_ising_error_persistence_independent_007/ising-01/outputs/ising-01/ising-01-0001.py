import subprocess, sys
print(sys.version)
for c in ["gcc --version", "cc --version", "python -c 'import numba; print(numba.__version__)'"]:
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(c, "->", r.stdout[:200], r.stderr[:200])
