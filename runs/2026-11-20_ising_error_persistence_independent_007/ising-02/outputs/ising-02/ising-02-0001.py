import importlib, time, os
for m in ["numba","numpy","scipy"]:
    try:
        importlib.import_module(m); print(m,"ok")
    except Exception as e: print(m,"missing",e)
import numpy as np
print(np.__version__, os.cpu_count())
t=time.time(); a=np.random.default_rng(1).random((256,256)); print(time.time()-t)