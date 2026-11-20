
import numpy as np
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/T{T:.3f}.npz')
    out=[]
    for sl in [slice(0,100),slice(100,300),slice(300,600)]:
        m2=d['bm2'][sl].mean(0); m4=d['bm4'][sl].mean(0)
        out.append(1-m4.mean()/(3*m2.mean()**2))
    print(T, np.round(out,5))
