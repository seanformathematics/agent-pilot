import json
d=json.load(open('/work/part1.json'))
for k,v in d.items(): print(k, repr(v['U']), repr(v['err']))
d=json.load(open('/work/part2.json'))
for k,v in d.items(): print(k, repr(v['U']), repr(v['err']))
