import shutil, importlib.util, os, platform
print(platform.python_version(), os.cpu_count())
print("gcc", shutil.which("gcc"), shutil.which("cc"), shutil.which("clang"))
print("numba", importlib.util.find_spec("numba") is not None)
import numpy; print(numpy.__version__)