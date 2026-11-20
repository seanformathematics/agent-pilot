
import subprocess
res=[]
for T,seed in [(1.9,201),(2.0,202),(2.1,203)]:
    out=subprocess.run(['python','/work/sim.py',str(T),str(seed),'2000','3000','6000'],capture_output=True,text=True,timeout=190)
    print(out.stdout.strip(),out.stderr[-500:])
