
import subprocess
for T in ['2.1','2.2']:
    r=subprocess.run(['python3','/work/sim.py',T,'250'],capture_output=True,text=True,timeout=280)
    print(r.stdout.strip(), r.stderr[-1000:])
