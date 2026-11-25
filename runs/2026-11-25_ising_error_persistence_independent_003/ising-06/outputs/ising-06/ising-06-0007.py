
import subprocess
out=[]
for T,seed in [(1.9,201),(2.0,202),(2.1,203)]:
    r=subprocess.run(['python3','/work/ising2.py',str(T),'2000','2000','4000',str(seed)],capture_output=True,text=True,timeout=190)
    out.append(r.stdout+r.stderr[-800:])
print("\n".join(out))
