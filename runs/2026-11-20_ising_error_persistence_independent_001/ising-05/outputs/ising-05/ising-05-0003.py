import subprocess, sys
driver = r'''
import sys, numpy as np, time
sys.path.insert(0,'/work')
from wolff import run
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
n=int(sys.argv[1]); seedbase=int(sys.argv[2]); tag=sys.argv[3]
for k,T in enumerate(Ts):
    t0=time.time()
    m2,m4=run(16,T,n,seedbase+k,5000)
    np.savez(f'/work/data_{tag}_{T}.npz',m2=m2,m4=m4)
    print(T, time.time()-t0, flush=True)
'''
open('/work/driver.py','w').write(driver)
subprocess.Popen('nohup python3 /work/driver.py 1000000 1000 A > /work/logA.txt 2>&1 &', shell=True)
print("started")
