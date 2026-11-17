import json
r = json.load(open('/work/results.json'))
for k in r:
    d = r[k]
    print(repr(d['T']), repr(d['U']), repr(d['err_chain_jack']), repr(d['err_block_jack']), d['seed'], min(d['block_U']), max(d['block_U']))
