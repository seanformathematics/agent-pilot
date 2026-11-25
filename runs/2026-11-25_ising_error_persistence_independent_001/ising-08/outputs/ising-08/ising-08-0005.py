import subprocess
out=[]
for k,T in enumerate(["1.9","2.0","2.1","2.2","2.4"]):
    r=subprocess.run(["python3","/work/ising.py",T,str(200+k),"1000","8000"],capture_output=True,text=True,timeout=280,cwd="/work")
    out.append(r.stdout+r.stderr[-500:])
print("\n".join(out))