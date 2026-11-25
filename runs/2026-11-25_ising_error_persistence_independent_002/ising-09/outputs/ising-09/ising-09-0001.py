import subprocess,sys
print(subprocess.run("which gcc cc clang; python3 -c 'import numba' ; nproc; free -m",shell=True,capture_output=True,text=True))
