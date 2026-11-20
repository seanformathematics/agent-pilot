import importlib, shutil, os
for m in ["numba","numpy","scipy"]:
    try:
        mod=importlib.import_module(m); print(m, getattr(mod,"__version__","?"))
    except Exception as e: print(m, "MISSING", e)
print("gcc", shutil.which("gcc"), shutil.which("cc"))
print(os.cpu_count())