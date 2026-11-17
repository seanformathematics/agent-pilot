import json
r=json.load(open("/work/results.json"))
for k,v in r.items(): print(k, repr(v["U"]), repr(v["err"]))
