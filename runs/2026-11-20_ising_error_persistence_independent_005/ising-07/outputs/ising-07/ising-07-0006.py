
import subprocess
for T,s in [(1.9,201),(2.4,202)]:
    r=subprocess.run(['python3','/work/wolff.py',str(T),str(s),'30'],capture_output=True,text=True,cwd='/work')
    print(r.stdout.strip(), r.stderr[-300:])
