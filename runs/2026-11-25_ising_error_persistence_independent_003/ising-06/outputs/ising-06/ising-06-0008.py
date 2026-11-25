
import subprocess
out=[]
for T,seed in [(2.2,204),(2.3,205),(2.4,206)]:
    r=subprocess.run(['python3','/work/ising2.py',str(T),'2000','2000','4000',str(seed)],capture_output=True,text=True,timeout=190)
    out.append(r.stdout+r.stderr[-800:])
print("\n".join(out))
