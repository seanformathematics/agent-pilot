import sys, os, platform
print(sys.version)
print(platform.platform())
print("cpus:", os.cpu_count())
try:
    import numpy as np
    print("numpy", np.__version__)
except Exception as e:
    print("numpy err", e)
try:
    import scipy
    print("scipy", scipy.__version__)
except Exception as e:
    print("scipy err", e)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("numba not available:", e)
print(os.getcwd(), os.listdir('/work') if os.path.exists('/work') else 'no /work')
