
import subprocess
r=subprocess.run('cd /work && python3 run.py 2.3 2301',shell=True,capture_output=True,text=True,timeout=590)
print(r.stdout,r.stderr)
