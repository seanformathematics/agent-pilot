import subprocess,sys
print(sys.version)
print(subprocess.run("which gcc cc clang; nproc; python -c 'import numba' ",shell=True,capture_output=True,text=True))
