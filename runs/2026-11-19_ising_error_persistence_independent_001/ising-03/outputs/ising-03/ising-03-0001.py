import importlib,sys,time
print(sys.version)
for m in ['numba','numpy','scipy']:
    try:
        mod=importlib.import_module(m); print(m, mod.__version__)
    except Exception as e: print(m,'missing',e)
