
import subprocess
r = subprocess.run(['python3','/work/ising.py','2.4,2.5,2.6,2.7,2.8','250000','0,1,2,3'],capture_output=True,text=True,timeout=580,cwd='/work')
print(r.stdout, r.stderr[-2000:])
