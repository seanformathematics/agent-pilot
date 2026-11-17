import sys, platform, shutil, os
print(sys.version)
print(platform.platform())
try:
    import numpy as np
    print('numpy', np.__version__)
except Exception as e:
    print('no numpy', e)
try:
    import numba
    print('numba', numba.__version__)
except Exception as e:
    print('no numba', repr(e))
for c in ['gcc', 'cc', 'clang', 'tcc']:
    print(c, shutil.which(c))
print('cpu_count', os.cpu_count())
os.makedirs('/work', exist_ok=True)
print(os.listdir('/work'))
