import importlib, time, sys
print(sys.version)
for m in ["numba","numpy","scipy"]:
    try:
        importlib.import_module(m); print(m,"ok")
    except Exception as e: print(m,"missing",e)
import numpy as np
t=time.time(); a=np.random.default_rng(1).random(10**7).sum(); print("t",time.time()-t)
