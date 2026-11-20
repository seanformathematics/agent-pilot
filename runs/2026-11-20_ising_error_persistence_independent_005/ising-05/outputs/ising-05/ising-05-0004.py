
import subprocess, json, time
Ts=['1.9','2.0','2.1','2.2','2.3','2.4']
res=[]
for k,T in enumerate(Ts):
    out=subprocess.run(['python3','/work/ising.py',T,'1000','3000','3000',str(201+k)],capture_output=True,text=True,timeout=280)
    r=json.loads(out.stdout.strip().splitlines()[-1]); res.append(r); print(r['T'],r['U'],r['Uerr'],r['Uerr_groups'])
json.dump(res,open('/work/check.json','w'))
