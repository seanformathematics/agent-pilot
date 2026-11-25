
import subprocess, time
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
t0=time.time()
for k,T in enumerate(Ts):
    r=subprocess.run(['python3','/work/ising_meta.py',str(T),'1024','3000','500',str(100+k),f'/work/pilot_{T}.npz'],capture_output=True,text=True)
    print(T,r.stdout.strip(),r.stderr[-300:])
print('total',time.time()-t0)
import numpy as np
for T in Ts:
    d=np.load(f'/work/pilot_{T}.npz')
    s2=d['sum2']/d['cnt']; s4=d['sum4']/d['cnt']
    U=1-s4.mean()/(3*s2.mean()**2)
    # jackknife over replica groups
    G=32; idx=np.arange(1024)%G
    Us=[]
    for g in range(G):
        keep=idx!=g
        a2=s2[keep].mean(); a4=s4[keep].mean()
        Us.append(1-a4/(3*a2*a2))
    Us=np.array(Us)
    err=np.sqrt((G-1)/G*np.sum((Us-Us.mean())**2))
    print(T, round(U,5), round(err,5))
