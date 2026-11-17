import importlib,os,time
for m in ['numba','numpy','scipy']:
    try:
        importlib.import_module(m); print(m,'ok')
    except Exception as e: print(m,'missing',e)
print(os.cpu_count())
