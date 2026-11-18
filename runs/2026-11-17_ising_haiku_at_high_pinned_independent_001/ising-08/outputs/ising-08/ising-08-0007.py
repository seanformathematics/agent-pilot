
import json
b1=json.load(open('/work/batch1.json'))
b2=json.load(open('/work/batch2.json'))
b3=json.load(open('/work/batch3.json'))
allr={}
for d in (b1,b2,b3):
    for k,v in d.items(): allr[float(k)]=(v['U'],v['err'])
for T in sorted(allr): print(T, repr(allr[T][0]), repr(allr[T][1]))
