
import json
d = {}
d.update(json.load(open('/work/results_part1.json')))
d.update(json.load(open('/work/results_part2.json')))
for T in sorted(d, key=float):
    r = d[T]
    print(T, repr(r['U']), repr(r['err']), r['nmeas'], r['seed'], r['sub100_std'], r['halfA'], r['halfB'])
