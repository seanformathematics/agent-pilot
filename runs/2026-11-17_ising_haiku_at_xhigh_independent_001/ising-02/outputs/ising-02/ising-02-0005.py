
import sys; sys.path.insert(0,'/work')
import numpy as np
from swlib import Ts, jk_U
d=np.load('/work/prod.npz')
S2=d['S2']; S4=d['S4']; n=int(d['n'])
reps=150
groups=[np.arange(k*reps,(k+1)*reps) for k in range(14)]
res=jk_U(S2/n,S4/n,groups)
for T,(u,e) in zip(Ts,res):
    print(repr(T), repr(float(u)), repr(float(e)))
print("n",n)
p=np.load('/work/pilot.npz')
res2=jk_U(p['S2']/int(p['n']),p['S4']/int(p['n']),[np.arange(k*100,(k+1)*100) for k in range(14)])
for T,(u,e) in zip(Ts,res2):
    print("pilot",T, repr(float(u)), repr(float(e)))
