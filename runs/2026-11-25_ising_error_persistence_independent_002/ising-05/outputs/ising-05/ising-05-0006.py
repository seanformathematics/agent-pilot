import subprocess
seeds={2.2:221,2.3:231,2.4:241}
out=""
for T,s in seeds.items():
    r=subprocess.run(["python","/work/sim.py",str(T),"1000","2000","20000",str(s)],capture_output=True,text=True)
    out+=r.stdout+r.stderr
print(out)
