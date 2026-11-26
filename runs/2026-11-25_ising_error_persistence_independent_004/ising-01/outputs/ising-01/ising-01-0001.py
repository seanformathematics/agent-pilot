import subprocess, sys
print(sys.version)
print(subprocess.run("which gcc cc clang; python -c 'import numba' ; nproc; free -m", shell=True, capture_output=True, text=True))
