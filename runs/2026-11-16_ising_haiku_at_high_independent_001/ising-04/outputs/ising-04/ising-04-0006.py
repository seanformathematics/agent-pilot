import json
r={}
r.update(json.load(open('/work/res_part1.json'))); r.update(json.load(open('/work/res_part2.json')))
for T in sorted(r, key=float): print(T, repr(r[T]['U']), repr(r[T]['err']), r[T]['n'])
