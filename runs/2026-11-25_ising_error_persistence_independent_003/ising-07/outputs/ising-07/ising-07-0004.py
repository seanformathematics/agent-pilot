
import subprocess
for T,seed in [('2.0',102),('2.1',103)]:
    r=subprocess.run(['python3','/work/ising.py',T,str(seed),'45000','5000'],capture_output=True,text=True,timeout=280)
    print(r.stdout.strip(), r.stderr[-500:])
