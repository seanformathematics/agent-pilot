import importlib,time,numpy as np
print(np.__version__)
for m in ['numba']:
    try:
        importlib.import_module(m); print(m,'ok')
    except Exception as e: print(m,'no',e)
t=time.time(); a=np.random.default_rng(1).random((1000,16,16)); 
for _ in range(100): b=np.exp(-a)
print('100 exp on 256k:',time.time()-t)