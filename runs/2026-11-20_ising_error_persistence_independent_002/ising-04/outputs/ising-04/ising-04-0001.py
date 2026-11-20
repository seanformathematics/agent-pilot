import importlib,sys,time
for m in ['numba','numpy','scipy']:
    try:
        importlib.import_module(m); print(m,'ok')
    except Exception as e: print(m,'missing',e)
print(sys.version)
import os; print(os.cpu_count())