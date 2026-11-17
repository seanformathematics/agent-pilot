import importlib,time,os
os.makedirs('/work',exist_ok=True)
for m in ['numba','numpy','scipy']:
    try:
        mod=importlib.import_module(m); print(m, getattr(mod,'__version__','?'))
    except Exception as e: print(m,'missing',e)
import numpy as np
t=time.time(); a=np.random.rand(256,16,16)
for _ in range(1000): b=a*a+a
print('1000 ops time',time.time()-t)
