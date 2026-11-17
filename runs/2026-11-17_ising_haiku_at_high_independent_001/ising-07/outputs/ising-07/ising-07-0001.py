import subprocess, sys
print(sys.version)
print(subprocess.run("which gcc cc clang; nproc; python3 -c 'import numba' ; python3 -c 'import numpy; print(numpy.__version__)'", shell=True, capture_output=True, text=True))