import shutil, importlib, os, sys
print(sys.version)
print("gcc", shutil.which("gcc"), shutil.which("cc"), shutil.which("clang"))
for m in ["numba","numpy","scipy","cython"]:
    try:
        importlib.import_module(m); print(m,"ok")
    except Exception as e: print(m,"no",e)
print(os.cpu_count())