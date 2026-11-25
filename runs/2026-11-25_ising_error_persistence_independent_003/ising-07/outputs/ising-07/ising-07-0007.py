
import subprocess
r=subprocess.run(['python3','/work/ising.py','2.3','105','45000','5000'],capture_output=True,text=True,timeout=580)
print(r.stdout.strip(), r.stderr[-500:])
