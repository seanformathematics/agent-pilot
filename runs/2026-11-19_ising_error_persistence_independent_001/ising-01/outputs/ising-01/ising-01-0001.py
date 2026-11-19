import subprocess,sys
print(sys.version)
for c in ["gcc --version","cc --version","python3 -c 'import numba'"]:
    r=subprocess.run(c,shell=True,capture_output=True,text=True)
    print(c, r.returncode, r.stdout[:200], r.stderr[:200])
import os; print(os.cpu_count())