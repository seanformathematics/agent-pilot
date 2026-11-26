
import random, math, time, json, os
os.makedirs('/work', exist_ok=True)
SCRIPT = r'''
import random, math, time, json, sys
N=16
T=float(sys.argv[1]); seed=int(sys.argv[2]); nclus=int(sys.argv[3]); out=sys.argv[4]
BIN=2000; EQ=20000
rng = random.Random(seed)
L=N*N
nb=[]
for i in range(L):
    x,y=i%N,i//N
    nb.append((((x+1)%N)+y*N, ((x-1)%N)+y*N, x+((y+1)%N)*N, x+((y-1)%N)*N))
s=[rng.choice((-1,1)) for _ in range(L)]
M=sum(s)
p=1-math.exp(-2.0/T)
rnd=rng.random
for _ in range(EQ):
    i=int(rnd()*L); sg=s[i]; s[i]=-sg; stack=[i]
    while stack:
        j=stack.pop()
        for k in nb[j]:
            if s[k]==sg and rnd()<p:
                s[k]=-sg; stack.append(k)
    # recompute M lazily below
M=sum(s)
bins=[]
s2=s4=0.0; cnt=0
t0=time.time()
for c in range(nclus):
    i=int(rnd()*L); sg=s[i]; s[i]=-sg; stack=[i]; size=1
    while stack:
        j=stack.pop()
        for k in nb[j]:
            if s[k]==sg and rnd()<p:
                s[k]=-sg; stack.append(k); size+=1
    M-=2*sg*size
    m2=M*M
    s2+=m2; s4+=m2*m2; cnt+=1
    if cnt==BIN:
        bins.append((s2/BIN/L/L, s4/BIN/L**2/L**2 if False else s4/BIN/(L*L)**2))
        s2=s4=0.0; cnt=0
res=dict(T=T,seed=seed,nclus=nclus,bins=bins,time=time.time()-t0)
json.dump(res,open(out,'w'))
print('done',T,seed,len(bins),time.time()-t0)
'''
open('/work/wolff.py','w').write(SCRIPT)
import subprocess
r=subprocess.run(['python3','/work/wolff.py','2.2','101','400000','/work/pilot_2.2_101.json'],capture_output=True,text=True,timeout=300)
print(r.stdout,r.stderr)
import json
import numpy as np
d=json.load(open('/work/pilot_2.2_101.json'))
b=np.array(d['bins'])
m2=b[:,0]; m4=b[:,1]
def U(m2,m4): return 1-m4.mean()/(3*m2.mean()**2)
# jackknife over bins
nb_=len(b)
Us=np.array([U(np.delete(m2,k),np.delete(m4,k)) for k in range(nb_)])
err=np.sqrt((nb_-1)/nb_*np.sum((Us-Us.mean())**2))
print('U',U(m2,m4),'jack err',err,'nbins',nb_)
