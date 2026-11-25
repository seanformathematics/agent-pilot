
import subprocess, json, numpy as np
out=[]
for T in ['1.9','2.0']:
    r=subprocess.run(['python3','/work/sim.py',T,'250'],capture_output=True,text=True,timeout=400)
    print(r.stdout.strip(), r.stderr[-1000:])
    d=np.load(f'/work/state_T{float(T):.2f}.npz')
    b=d['blocks']; print('first blocks |m|:', np.round(b[:6],4), 'n blocks', len(b))
