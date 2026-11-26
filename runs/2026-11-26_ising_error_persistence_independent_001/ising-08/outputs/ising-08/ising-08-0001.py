import shutil, importlib.util, os
print(shutil.which('gcc'), shutil.which('cc'), shutil.which('clang'))
print(importlib.util.find_spec('numba') is not None)
import numpy; print(numpy.__version__, os.cpu_count())