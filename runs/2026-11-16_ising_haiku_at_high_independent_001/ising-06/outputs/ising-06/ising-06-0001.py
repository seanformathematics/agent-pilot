import importlib, time, os
for m in ["numba","numpy","scipy"]:
    try:
        importlib.import_module(m); print(m,"ok")
    except Exception as e: print(m,"missing",e)
import numpy as np
print(np.__version__, os.cpu_count())
t=time.time(); a=np.random.rand(2000,16,16); 
for _ in range(100): b=np.roll(a,1,axis=1)+a
print("100 ops",time.time()-t)