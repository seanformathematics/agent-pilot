
import subprocess, time
t0=time.time()
for T in ['2.2','2.3','2.4']:
    for sd in [301,302,303]:
        r=subprocess.run(['python3','/work/wolff.py',T,str(sd),'400000',f'/work/res_{T}_{sd}.json'],capture_output=True,text=True,timeout=400)
        print(r.stdout.strip(), r.stderr.strip()[-300:])
print('elapsed',time.time()-t0)
