import json
r=json.load(open('/work/res.json'))
for T,v in r.items(): print(T, repr(v['U']), repr(v['err']), v['seed'], v['start'])