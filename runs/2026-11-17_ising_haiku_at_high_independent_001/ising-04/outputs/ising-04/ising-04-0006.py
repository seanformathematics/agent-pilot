
import json, numpy as np
r1=json.load(open('/work/run1.json')); r2=json.load(open('/work/run2.json'))
for T in r1:
    U1,e1=r1[T]['U'],r1[T]['Uerr']; U2,e2=r2[T]['U'],r2[T]['Uerr']
    U=(U1+U2)/2; e=np.sqrt(e1**2+e2**2)/2
    print(T, repr(U), repr(e), abs(U1-U2)/np.sqrt(e1**2+e2**2+1e-30))
