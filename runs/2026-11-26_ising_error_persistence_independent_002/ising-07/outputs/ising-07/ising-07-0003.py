
import os,shutil,subprocess
print(subprocess.run(['df','-h','/work','/tmp','/dev/shm'],capture_output=True,text=True).stdout)
for f in os.listdir('/work'): print(f, os.path.getsize('/work/'+f))
for f in os.listdir('/work'):
    os.remove('/work/'+f)
print(subprocess.run(['df','-h','/work'],capture_output=True,text=True).stdout)
