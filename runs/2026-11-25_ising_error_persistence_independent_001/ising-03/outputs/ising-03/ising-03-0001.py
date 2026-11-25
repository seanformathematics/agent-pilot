import sys, importlib, os
print(sys.version)
for m in ["numba","numpy","scipy"]:
    try:
        importlib.import_module(m); print(m,"ok")
    except Exception as e: print(m,e)
print(os.cpu_count())