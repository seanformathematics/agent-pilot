
import sys, numpy as np
sys.path.insert(0,'/work')
import wolff
def U_of(b2,b4): return 1-b4.mean()/(3*b2.mean()**2)
for T in [2.4,2.3]:
    Us=[]
    for sd in range(20):
        s,M,b2,b4=wolff.run(T, 9000+sd, 200000, 100)
        Us.append(U_of(b2,b4))
    Us=np.array(Us)
    print(T,'mean',Us.mean(),'sem',Us.std(ddof=1)/np.sqrt(len(Us)),'std',Us.std(ddof=1))
    print(np.round(np.sort(Us),5))
