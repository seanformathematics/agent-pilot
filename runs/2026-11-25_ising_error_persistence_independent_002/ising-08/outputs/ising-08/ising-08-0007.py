
import subprocess
r=subprocess.run(['python3','/work/sim.py','2.4','250'],capture_output=True,text=True,timeout=280)
print(r.stdout.strip(), r.stderr[-1000:])
