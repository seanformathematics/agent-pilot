import importlib, time, sys
print(sys.version)
for m in ["numba","numpy","scipy"]:
    try:
        importlib.import_module(m); print(m,"ok")
    except Exception as e: print(m,"missing",e)
import os; print(os.cpu_count())