import subprocess,sys
print(sys.version)
for m in ["numba","numpy","scipy"]:
    try:
        __import__(m); print(m,"ok")
    except Exception as e: print(m,e)
print(subprocess.run("which gcc cc clang; nproc; cat /proc/cpuinfo | grep 'model name' | head -1",shell=True,capture_output=True,text=True).stdout)