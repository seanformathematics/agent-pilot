
import subprocess
for T in ['2.1','2.2']:
    r=subprocess.run(['python3','/work/run_ising.py',T,'280','13000','3000','202'],capture_output=True,text=True,timeout=290)
    print(r.stdout,r.stderr[-1000:])
