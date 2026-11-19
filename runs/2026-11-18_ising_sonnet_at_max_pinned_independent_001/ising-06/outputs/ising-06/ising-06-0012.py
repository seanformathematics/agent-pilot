
import json
d=json.load(open('/work/final_results.json'))
for p in d['points']:
    print(f"{{\"T\": {p['T']}, \"status\": \"ok\", \"U\": {p['U']!r}, \"U_err\": {p['U_err']!r}, \"n_samples\": {p['n_samples']}}},")
print(len(d['seeds']), d['seeds'][:4], d['seeds'][-4:])
print(max(p['U_err'] for p in d['points']))
