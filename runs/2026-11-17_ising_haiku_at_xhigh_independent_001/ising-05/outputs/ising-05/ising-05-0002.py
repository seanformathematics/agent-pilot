import shutil, subprocess
for c in ["gcc","cc","clang","tcc"]:
    print(c, shutil.which(c))
try:
    print(subprocess.run(["gcc","--version"],capture_output=True,text=True).stdout)
except Exception as e: print(e)
