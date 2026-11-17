import sys, shutil, os
print(sys.version)
for c in ['gcc', 'cc', 'clang']:
    print(c, shutil.which(c))
try:
    import numba
    print('numba', numba.__version__)
except Exception as e:
    print('no numba:', repr(e))
print('cpu_count', os.cpu_count())
