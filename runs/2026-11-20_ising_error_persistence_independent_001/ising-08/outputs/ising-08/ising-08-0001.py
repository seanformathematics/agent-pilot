import importlib, time, sys
print(sys.version)
for m in ["numba","numpy","scipy"]:
    try:
        importlib.import_module(m); print(m,"ok")
    except Exception as e: print(m,"missing",e)
import numpy as np
t=time.time(); s=0
for i in range(10**7): s+=i
print("loop 1e7 time", time.time()-t)
