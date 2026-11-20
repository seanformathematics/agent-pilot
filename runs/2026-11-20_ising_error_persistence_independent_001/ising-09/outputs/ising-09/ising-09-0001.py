import subprocess,sys
print(sys.version)
print(subprocess.run("which gcc cc clang; nproc; python3 -c 'import numba' ; free -m",shell=True,capture_output=True,text=True))