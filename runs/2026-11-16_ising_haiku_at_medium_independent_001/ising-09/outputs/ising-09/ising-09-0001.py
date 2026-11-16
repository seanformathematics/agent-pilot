import shutil,subprocess,importlib.util
print(shutil.which('gcc'),shutil.which('cc'),shutil.which('clang'))
print(importlib.util.find_spec('numba') is not None)
import os;print(os.cpu_count())