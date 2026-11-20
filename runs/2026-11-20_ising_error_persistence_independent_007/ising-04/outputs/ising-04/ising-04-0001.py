import importlib, time, numpy as np
for m in ['numba','scipy']:
    try:
        importlib.import_module(m); print(m,'yes')
    except Exception as e: print(m,'no',e)
print(np.__version__)
t=time.time(); a=np.random.rand(10**7); print(time.time()-t)