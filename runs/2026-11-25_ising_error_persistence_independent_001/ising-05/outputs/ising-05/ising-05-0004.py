
import subprocess
r=subprocess.run(['python3','/work/run_ising.py','2.269','450','20000','3000','101'],capture_output=True,text=True,timeout=590)
print(r.stdout,r.stderr[-2000:])
