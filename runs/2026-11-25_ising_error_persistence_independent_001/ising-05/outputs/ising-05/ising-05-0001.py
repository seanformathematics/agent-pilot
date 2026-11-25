import importlib,time,os
for m in ['numba','numpy','scipy']:
    try:
        importlib.import_module(m); print(m,'ok')
    except Exception as e: print(m,'missing',e)
import numpy as np
print(os.cpu_count(), np.__version__)
t=time.time(); a=np.random.default_rng(1).random((1000,256)); 
for _ in range(1000): b=a*a
print('1000 ops on 256k arrays', time.time()-t)