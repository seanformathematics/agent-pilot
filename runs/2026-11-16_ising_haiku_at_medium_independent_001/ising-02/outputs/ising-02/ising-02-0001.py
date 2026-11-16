import importlib,time
for m in ["numba","numpy"]:
    try:
        importlib.import_module(m); print(m,"ok")
    except Exception as e: print(m,e)
