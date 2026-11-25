import importlib, shutil, os
for m in ["numba","numpy","scipy"]:
    try:
        importlib.import_module(m); print(m,"ok")
    except Exception as e: print(m,"missing",e)
print("gcc", shutil.which("gcc"), "cc", shutil.which("cc"))
os.makedirs("/work",exist_ok=True)
print(os.cpu_count())