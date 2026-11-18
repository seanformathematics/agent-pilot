import sys, os, shutil, platform
print("python", sys.version)
print("platform", platform.platform())
print("cpu count", os.cpu_count())
for mod in ["numpy", "scipy", "numba", "cython"]:
    try:
        m = __import__(mod)
        print(mod, getattr(m, "__version__", "?"))
    except Exception as e:
        print(mod, "NOT AVAILABLE:", e)
for exe in ["gcc", "cc", "clang", "tcc"]:
    print(exe, shutil.which(exe))
print(os.getcwd(), os.path.exists("/work"))
