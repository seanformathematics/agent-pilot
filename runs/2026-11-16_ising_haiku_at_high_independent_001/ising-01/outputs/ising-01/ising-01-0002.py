import shutil,subprocess
for c in ["gcc","cc","clang","tcc"]:
    print(c, shutil.which(c))
print(subprocess.run(["bash","-lc","gcc --version | head -1; nproc"],capture_output=True,text=True).stdout)
