
import subprocess, numpy as np, time
def U_of(S2,S4): return 1-np.mean(S4)/(3*np.mean(S2)**2)
for T,seed in [('2.1',103),('2.2',104)]:
    r=subprocess.run(['python3','/work/ising.py',T,str(seed),'1000','4000','4000',f'/work/prod_{T}.npz'],capture_output=True,text=True)
    print(r.stdout.strip(),r.stderr[-300:])
    d=np.load(f'/work/prod_{T}.npz'); S2,S4=d['S2'],d['S4']; R=len(S2)
    U=U_of(S2,S4); jk=np.array([U_of(np.delete(S2,i),np.delete(S4,i)) for i in range(R)])
    print(T,'U',U,'err',np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2)))
