import json, math
final = json.load(open('/work/final_pooled.json'))
prod = {round(r['T'],2): r for r in [json.load(open(f'/work/res_T{T:.2f}.json')) for T in [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]]}
out = []
for f in final:
    T = round(f['T'], 2)
    print(repr(T), repr(f['U']), repr(f['U_err']), f['n_samples'], f['n_chains'], f['U_err_chain'])
    out.append(dict(T=T, U=f['U'], U_err=f['U_err'], n=f['n_samples'], seeds=f['seeds']))
allseeds = sorted(set(s for o in out for s in o['seeds']))
print("ALL SEEDS:", allseeds)
print("count:", len(allseeds))
print("U<2/3 all:", all(o['U'] < 2/3 for o in out), " max U_err:", max(o['U_err'] for o in out))
