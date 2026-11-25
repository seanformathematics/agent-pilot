import importlib,sys,os
print(sys.version, os.cpu_count())
for m in ['numba','numpy','scipy']:
    try:
        importlib.import_module(m); print(m,'ok')
    except Exception as e: print(m,'missing',e)