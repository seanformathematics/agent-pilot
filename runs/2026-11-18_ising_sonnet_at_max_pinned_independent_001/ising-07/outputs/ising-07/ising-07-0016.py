
import json
d=json.load(open('/work/final_results.json'))
for p in d['points']:
    print('{"T": %s, "status": "ok", "U": %.8f, "U_err": %.4e, "n_samples": %d},'%(p['T'],p['U'],p['U_err'],p['n_samples']))
print("K,R,neq,steps:",d['K'],d['R'],d['neq'],d['steps'])
print("seeds:",d['seeds'])
print("distinct seeds:",len(set(d['seeds'])))
