
import json, math
r1=json.load(open('/work/run1.json')); r2=json.load(open('/work/run2.json'))
for T in ['1.9','2.0','2.1','2.2','2.3','2.4']:
    u1,e1=r1[T]['U'],r1[T]['Uerr']; u2,e2=r2[T][0],r2[T][1]
    U=(u1+u2)/2; E=math.sqrt(e1**2+e2**2)/2
    print(T, repr(U), repr(E), abs(u1-u2)/math.sqrt(e1**2+e2**2+1e-30))
