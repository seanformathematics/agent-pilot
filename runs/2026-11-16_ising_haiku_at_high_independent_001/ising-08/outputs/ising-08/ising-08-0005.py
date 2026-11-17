
import json
for f in ['/work/res_batch1.json','/work/res_batch2.json']:
    d=json.load(open(f))
    for T,v in d.items(): print(T, repr(v['U']), repr(v['err']), v['seed'])
