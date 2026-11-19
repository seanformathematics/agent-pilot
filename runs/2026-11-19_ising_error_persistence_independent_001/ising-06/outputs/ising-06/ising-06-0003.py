
import subprocess, numpy as np, time
t=time.time()
r=subprocess.run(['python3','/work/ising.py','2.3','11','1000','2000','2000','/work/pilot_2.3.npz'],capture_output=True,text=True)
print(r.stdout,r.stderr,time.time()-t)
d=np.load('/work/pilot_2.3.npz')
S2,S4=d['S2'],d['S4']
def U_of(S2,S4): return 1-np.mean(S4)/(3*np.mean(S2)**2)
U=U_of(S2,S4)
R=len(S2)
jk=np.array([U_of(np.delete(S2,i),np.delete(S4,i)) for i in range(R)])
err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
print('U',U,'jk err',err)
