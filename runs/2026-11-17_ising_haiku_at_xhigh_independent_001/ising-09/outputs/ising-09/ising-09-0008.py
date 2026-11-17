import json
r1 = json.load(open('/work/results_part1.json'))
r2 = json.load(open('/work/results_part2.json'))
allr = {**r1, **r2}
for T in sorted(allr, key=float):
    d = allr[T]
    print(T, repr(d['U']), repr(d['Uerr']), d['n_meas'], d['seed'], repr(d['U_first_half']), repr(d['U_second_half']))
