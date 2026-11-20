
import subprocess
for T,seed in [(1.9,101),(2.0,102),(2.1,103)]:
    r=subprocess.run(['python3','/work/ising.py',str(T),'400','50000','5000',str(seed),f'/work/run_{T}.npz'],capture_output=True,text=True,timeout=280)
    print(r.stdout,r.stderr[-500:])
