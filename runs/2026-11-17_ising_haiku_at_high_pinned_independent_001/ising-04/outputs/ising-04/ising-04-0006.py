
import subprocess
r = subprocess.run(['python3','/work/ising.py','1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2','250000','0,1,2,3'],capture_output=True,text=True,timeout=580,cwd='/work')
print(r.stdout, r.stderr[-2000:])
