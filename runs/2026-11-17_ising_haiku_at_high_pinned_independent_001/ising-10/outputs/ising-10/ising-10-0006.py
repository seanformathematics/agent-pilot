
import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
for T in Ts:
    d=np.load(f'/work/res_{T}.npz')
    m2b,m4b=d['m2b'],d['m4b']
    n=m2b.shape[1]
    out=[]
    for sl in [slice(0,n//2),slice(n//2,n)]:
        a2=m2b[:,sl].mean(); a4=m4b[:,sl].mean()
        out.append(1-a4/(3*a2*a2))
    print(T, round(out[0],5), round(out[1],5), "diff", round(out[1]-out[0],5))
