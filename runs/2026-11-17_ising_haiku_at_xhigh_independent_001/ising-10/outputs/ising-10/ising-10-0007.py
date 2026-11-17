import json
p = {}
for fn in ['/work/prod_part1.json', '/work/prod_part2.json']:
    p.update({float(k): v for k, v in json.load(open(fn)).items()})
v = {float(k): d for k, d in json.load(open('/work/verify.json')).items()}
for T in sorted(p):
    d = p[T]
    print(T, repr(d['U']), repr(d['err']), d['seed'], d['U_half1'], d['U_half2'], '| verify', repr(v[T]['U']), repr(v[T]['err']), v[T]['seed'])
