import shutil, importlib.util, time, sys
print(sys.version)
print("gcc", shutil.which("gcc"), shutil.which("cc"))
print("numba", importlib.util.find_spec("numba") is not None)
import numpy as np; print(np.__version__)
