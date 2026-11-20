
import subprocess
for T,seed in [(2.2,204),(2.4,205)]:
    out=subprocess.run(['python','/work/sim.py',str(T),str(seed),'2000','3000','6000'],capture_output=True,text=True,timeout=280)
    print(out.stdout.strip(),out.stderr[-500:])
