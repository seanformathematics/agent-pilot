import subprocess, sys, os, platform
print(sys.version)
print(platform.platform())
print(os.cpu_count())
for cmd in ["which gcc cc clang tcc", "gcc --version", "python3 -c 'import numba; print(numba.__version__)'", "pip list 2>/dev/null | head -100"]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("CMD:", cmd, "\nOUT:", r.stdout, "\nERR:", r.stderr[:500])
