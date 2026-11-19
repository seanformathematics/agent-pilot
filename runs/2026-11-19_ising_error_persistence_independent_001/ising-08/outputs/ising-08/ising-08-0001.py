import subprocess,sys
print(sys.version)
for c in ["gcc --version","cc --version","nproc"]:
    try:
        print(subprocess.run(c,shell=True,capture_output=True,text=True,timeout=20).stdout[:200])
    except Exception as e: print(e)
try:
    import numba; print("numba",numba.__version__)
except Exception as e: print("no numba",e)
