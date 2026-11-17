import sys, os, subprocess, platform
print("python", sys.version)
print("platform", platform.platform())
print("cpus", os.cpu_count())
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba:", e)
try:
    import numpy
    print("numpy", numpy.__version__)
except Exception as e:
    print("no numpy", e)
try:
    import scipy
    print("scipy", scipy.__version__)
except Exception as e:
    print("no scipy", e)
for cmd in (["gcc", "--version"], ["cc", "--version"], ["clang", "--version"]):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        print(cmd[0], out.returncode, out.stdout.splitlines()[:1])
    except Exception as e:
        print(cmd[0], "not available:", e)
print(open("/proc/meminfo").read().splitlines()[:2])
print("cwd", os.getcwd(), os.listdir("/work") if os.path.exists("/work") else "no /work")
