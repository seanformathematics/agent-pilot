import subprocess, sys, importlib
print(sys.version)
for m in ["numba","numpy","scipy","cython"]:
    try:
        importlib.import_module(m); print(m,"ok")
    except Exception as e: print(m,"missing",e)
for c in ["gcc","cc","clang"]:
    r=subprocess.run(["which",c],capture_output=True,text=True); print(c, r.stdout.strip(), r.returncode)
import os; print(os.cpu_count())
