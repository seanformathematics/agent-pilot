import os,glob,shutil
for f in glob.glob('/work/*'):
    print(f, os.path.getsize(f))
    os.remove(f)
print(shutil.disk_usage('/work'))