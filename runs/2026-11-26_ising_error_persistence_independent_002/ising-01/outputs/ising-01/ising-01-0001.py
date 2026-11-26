import subprocess,sys,os
print(sys.version)
for m in ['numba','numpy','scipy']:
    try:
        __import__(m); print(m,'ok')
    except Exception as e: print(m,e)
print(subprocess.run('which gcc cc clang; nproc; python3 -c "import os;print(os.cpu_count())"',shell=True,capture_output=True,text=True).stdout)