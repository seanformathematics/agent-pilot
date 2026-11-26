import shutil,subprocess,os
print(shutil.which('gcc'),shutil.which('cc'),shutil.which('clang'))
try:
    import numba; print('numba',numba.__version__)
except Exception as e: print('no numba',e)
import numpy; print(numpy.__version__, os.cpu_count())