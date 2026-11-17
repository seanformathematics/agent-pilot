
import json
d=json.load(open('/work/res1.json'))
for k,v in d.items(): print(k, repr(v[0]), repr(v[1]))
