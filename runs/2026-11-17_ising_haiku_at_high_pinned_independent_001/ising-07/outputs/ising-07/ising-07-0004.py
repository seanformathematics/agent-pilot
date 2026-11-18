import json
d=json.load(open('/work/prod1.json'))
for T,v in d.items(): print(T, repr(v['U']), repr(v['err']))
