import subprocess
out=""
for T,s in [(2.3,901),(1.9,902)]:
    r=subprocess.run(["python","/work/sim.py",str(T),"500","50","4000",str(s)],capture_output=True,text=True)
    out+=r.stdout+r.stderr
print(out)
