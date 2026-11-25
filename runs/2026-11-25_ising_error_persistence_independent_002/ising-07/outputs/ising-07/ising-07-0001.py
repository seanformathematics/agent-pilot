import subprocess,shutil,os
print(shutil.which('gcc'), shutil.which('cc'), shutil.which('clang'))
try:
    import numba; print('numba', numba.__version__)
except Exception as e: print('no numba', e)
import multiprocessing; print(multiprocessing.cpu_count())
os.makedirs('/work',exist_ok=True)
