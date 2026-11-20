import importlib, time, numpy as np
for m in ["numba","scipy"]:
    try:
        importlib.import_module(m); print(m,"ok")
    except Exception as e: print(m,"missing",e)
print(np.__version__)
t=time.time(); a=np.random.default_rng(1).random(10**7); print(time.time()-t)
