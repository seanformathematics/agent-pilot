import subprocess
seeds={1.9:191,2.0:201,2.1:211}
out=""
for T,s in seeds.items():
    r=subprocess.run(["python","/work/sim.py",str(T),"1000","2000","20000",str(s)],capture_output=True,text=True)
    out+=r.stdout+r.stderr
print(out)
