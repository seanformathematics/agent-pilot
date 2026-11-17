import subprocess,sys,time
print(sys.version)
for c in ["gcc --version","cc --version","python -c 'import numba'"]:
    r=subprocess.run(c,shell=True,capture_output=True,text=True);print(c,r.returncode,r.stdout[:200],r.stderr[:200])
import numpy as np;print(np.__version__)
t=time.time();x=np.random.rand(10**7);print(time.time()-t)