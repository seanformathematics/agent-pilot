import sys, platform, shutil, os
print(sys.version)
print(platform.platform())
print("cpu_count:", os.cpu_count())
for tool in ['gcc', 'cc', 'clang', 'tcc']:
    print(tool, shutil.which(tool))
try:
    import numba
    print('numba', numba.__version__)
except Exception as e:
    print('no numba:', e)
import numpy, scipy
print('numpy', numpy.__version__, 'scipy', scipy.__version__)
print('cwd', os.getcwd(), os.path.exists('/work'), os.access('/work', os.W_OK))
