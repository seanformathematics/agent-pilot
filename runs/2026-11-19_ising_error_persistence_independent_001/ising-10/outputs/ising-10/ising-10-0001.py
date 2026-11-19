import subprocess,sys,time
print(sys.version)
print(subprocess.run("which gcc cc clang; gcc --version | head -1; python -c 'import numba' ; nproc",shell=True,capture_output=True,text=True))
import numpy as np
t=time.time(); a=np.random.default_rng(1).random(10**7); print(time.time()-t)