import sys, time, importlib
print(sys.version)
for m in ["numba","numpy","scipy"]:
    try:
        mod=importlib.import_module(m); print(m, getattr(mod,"__version__","?"))
    except Exception as e:
        print(m, "MISSING", e)
import os
print(os.cpu_count())
t=time.time()
s=0
for i in range(10**7): s+=i
print("loop 1e7:", time.time()-t)
