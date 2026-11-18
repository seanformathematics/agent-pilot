import sys, subprocess, os, time
print(sys.version)
for mod in ["numpy", "scipy", "numba"]:
    try:
        m = __import__(mod)
        print(mod, getattr(m, "__version__", "?"))
    except Exception as e:
        print(mod, "NOT AVAILABLE:", e)
for cmd in [["gcc", "--version"], ["cc", "--version"], ["clang", "--version"], ["nproc"]]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        print(cmd, out.returncode, out.stdout.splitlines()[:1])
    except Exception as e:
        print(cmd, "not available:", e)
print("cwd", os.getcwd(), os.listdir("."))
t = time.time()
s = 0
for i in range(10**7):
    s += i
print("pure python 1e7 loop s:", round(time.time() - t, 3))
