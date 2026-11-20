import sys, numpy as np
sys.path.insert(0,'/work')
from wolff import run
for T in [2.3,2.4]:
    Us=[]
    for s in range(6):
        m2,m4=run(16,T,500000,5000+s+int(T*100),3000)
        Us.append(1-m4.mean()/(3*m2.mean()**2))
    Us=np.array(Us)
    print(T, np.round(Us,5), 'mean',Us.mean().round(5),'std',Us.std(ddof=1).round(5), 'expected from 2e6 run ~ std*sqrt(4)/sqrt(6)')
