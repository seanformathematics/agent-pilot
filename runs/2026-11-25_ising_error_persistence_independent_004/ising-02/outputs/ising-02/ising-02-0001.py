import subprocess,sys
print(sys.version)
print(subprocess.run("which gcc cc clang; gcc --version | head -1; nproc; python -c 'import numba' ",shell=True,capture_output=True,text=True))
