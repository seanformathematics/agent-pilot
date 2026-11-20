import shutil, importlib.util, os
print(shutil.which('gcc'), shutil.which('cc'), shutil.which('clang'))
print(importlib.util.find_spec('numba'))
print(os.cpu_count())