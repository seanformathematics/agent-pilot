
import subprocess
temps=[1.9,2.0,2.1,2.2,2.3,2.4]
for k,T in enumerate(temps):
    r=subprocess.run(['python3','/work/wolff.py',str(T),str(101+k),'60'],capture_output=True,text=True,cwd='/work')
    print(r.stdout.strip(), r.stderr[-500:])
