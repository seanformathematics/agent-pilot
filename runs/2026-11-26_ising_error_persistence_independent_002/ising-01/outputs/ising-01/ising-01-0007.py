
import sys, numpy as np
sys.path.insert(0,'/work')
import wolff
def U_of(b2,b4): return 1-b4.mean()/(3*b2.mean()**2)
for T in [1.9,2.4]:
    Us=[]
    for sd in range(6):
        s,M,b2,b4=wolff.run(T, 7000+sd, 100000, 100)
        Us.append(U_of(b2,b4))
    Us=np.array(Us)
    print(T, 'mean',Us.mean(),'seed std',Us.std(ddof=1), 'expected from 3e5 jackknife ~ scaled to 1e5:', {1.9:1.244e-05,2.4:4.883e-04}[T]*np.sqrt(3))
