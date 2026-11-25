
script = r'''
import numpy as np, time, sys, json
N=16
T=float(sys.argv[1]); R=int(sys.argv[2]); neq=int(sys.argv[3]); nmeas=int(sys.argv[4]); seed=int(sys.argv[5])
rng=np.random.default_rng(seed)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0), ((ii+jj)%2==1)]
acc=np.zeros(9)
for d in [4,8]: acc[d]=np.exp(-d/T)
S=rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
def sweep(S):
    for m in masks:
        nn = (np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
        dE = 2*S*nn
        r = rng.random(S.shape)
        flip = m & ((dE<=0) | (r < acc[np.clip(dE,0,8)]))
        S[flip] *= -1
for k in range(neq): sweep(S)
s2=np.zeros(R); s4=np.zeros(R); se=np.zeros(R); sa=np.zeros(R)
# store per-sweep m^2 and m^4 per replica for block-based error checks
M2=np.zeros((nmeas,R)); M4=np.zeros((nmeas,R)); MA=np.zeros((nmeas,R))
t=time.time()
for k in range(nmeas):
    sweep(S)
    m=S.reshape(R,-1).mean(1)
    M2[k]=m*m; M4[k]=m**4; MA[k]=np.abs(m)
np.savez(f'/work/run_T{T}_N{N}_s{seed}.npz', M2=M2, M4=M4, MA=MA, T=T, R=R, neq=neq, seed=seed)
print("done T",T,"time",time.time()-t)
'''
open('/work/ising.py','w').write(script)
import subprocess
r=subprocess.run(['python3','/work/ising.py','2.2','2000','1000','2000','101'],capture_output=True,text=True,timeout=500)
print(r.stdout,r.stderr[-2000:])
