import shutil,subprocess
print(shutil.which('gcc'), shutil.which('cc'), shutil.which('clang'))
try:
    import numba; print('numba', numba.__version__)
except Exception as e: print('no numba', e)
import os; print(os.cpu_count())