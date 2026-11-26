
import subprocess,time
script=open('/work/prod.py').read()
out=[]
for T,seed in [(2.2,104),(2.3,105),(2.4,106)]:
    t=time.time()
    r=subprocess.run(["python","/work/prod.py",str(T),str(seed),"3000","4000"],capture_output=True,text=True,cwd="/work",timeout=400)
    out.append((T,r.returncode,r.stderr[-300:],round(time.time()-t,1)))
print(out)
