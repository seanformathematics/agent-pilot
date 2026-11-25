import importlib,time,sys
print(sys.version)
for m in ['numba','numpy','scipy']:
    try:
        importlib.import_module(m); print(m,'ok')
    except Exception as e: print(m,e)
import numpy as np
t=time.time(); a=np.random.rand(10**7); print(time.time()-t)