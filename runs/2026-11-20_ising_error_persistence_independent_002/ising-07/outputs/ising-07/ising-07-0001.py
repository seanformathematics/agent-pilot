import importlib, shutil, os
for m in ["numba","numpy","scipy","cython"]:
    try:
        importlib.import_module(m); print(m,"ok")
    except Exception as e: print(m,"missing",e)
print("gcc", shutil.which("gcc"), shutil.which("cc"), shutil.which("clang"))
print(os.cpu_count())