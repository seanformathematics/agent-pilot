import shutil,os,subprocess
print(shutil.which('gcc'),shutil.which('cc'),shutil.which('clang'))
try:
    import numba; print('numba',numba.__version__)
except Exception as e: print('no numba',e)
print(os.cpu_count())
import numpy; print(numpy.__version__)