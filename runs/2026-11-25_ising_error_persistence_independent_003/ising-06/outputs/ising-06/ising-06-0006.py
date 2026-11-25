
import os,glob,subprocess
for f in glob.glob('/work/*.npz'):
    os.remove(f)
print(subprocess.run(['df','-h','/work','/tmp'],capture_output=True,text=True).stdout)
script = r'''
import numpy as np, time, sys
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
S2=np.zeros(R); S4=np.zeros(R); SA=np.zeros(R)
# time-block sums: 10 blocks per replica to check drift
nb=10; blk=nmeas//nb
B2=np.zeros((nb,R)); B4=np.zeros((nb,R))
t=time.time()
for k in range(nmeas):
    sweep(S)
    m=S.reshape(R,-1).mean(1)
    m2=m*m
    S2+=m2; S4+=m2*m2; SA+=np.abs(m)
    B2[k//blk]+=m2; B4[k//blk]+=m2*m2
np.savez(f'/work/sums_T{T}_s{seed}.npz', S2=S2/nmeas, S4=S4/nmeas, SA=SA/nmeas, B2=B2/blk, B4=B4/blk, T=T, R=R, neq=neq, nmeas=nmeas, seed=seed)
print("done T",T,"time",round(time.time()-t,1))
'''
open('/work/ising2.py','w').write(script)
