
import subprocess, time
import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
t0=time.time()
for k,T in enumerate(Ts):
    r=subprocess.run(['python3','/work/ising_meta.py',str(T),'1024','4000','2000',str(500+k),f'/work/prod_{T}.npz'],capture_output=True,text=True)
    print(T,r.stderr[-300:])
print('total',time.time()-t0)
def UJ(s2,s4,G=32):
    n=len(s2); idx=np.arange(n)%G
    U=1-s4.mean()/(3*s2.mean()**2)
    Us=[]
    for g in range(G):
        k=idx!=g
        a2=s2[k].mean(); a4=s4[k].mean()
        Us.append(1-a4/(3*a2*a2))
    Us=np.array(Us)
    return U, np.sqrt((G-1)/G*np.sum((Us-Us.mean())**2))
for T in Ts:
    d=np.load(f'/work/prod_{T}.npz'); p=np.load(f'/work/pilot_{T}.npz')
    s2=d['sum2']/d['cnt']; s4=d['sum4']/d['cnt']
    U,e=UJ(s2,s4)
    Uc,ec=UJ(s2[:512],s4[:512]); Uh,eh=UJ(s2[512:],s4[512:])
    ps2=p['sum2']/p['cnt']; ps4=p['sum4']/p['cnt']
    Up,ep=UJ(ps2,ps4)
    print(f"T={T} prod U={U:.5f}±{e:.5f} | cold {Uc:.5f}±{ec:.5f} hot {Uh:.5f}±{eh:.5f} | pilot {Up:.5f}±{ep:.5f}")
