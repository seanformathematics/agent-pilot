import shutil,subprocess
print(shutil.which("gcc"),shutil.which("cc"),shutil.which("clang"))
print(subprocess.run(["bash","-lc","gcc --version | head -1; nproc"],capture_output=True,text=True).stdout)
