import time, importlib
for m in ["numba","numpy","scipy"]:
    try:
        importlib.import_module(m); print(m,"ok")
    except Exception as e: print(m,"missing",e)
import numpy as np
t=time.time(); a=np.random.default_rng(1).random((1000,16,16)); 
for _ in range(100): b=np.roll(a,1,axis=1)+a
print("100 roll-add on 1000x16x16:",time.time()-t)
