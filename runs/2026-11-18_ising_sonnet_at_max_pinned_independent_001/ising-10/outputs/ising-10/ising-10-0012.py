
src = open('/work/prod.py').read()
src2 = src.replace("SEED_BASE = 880000", "SEED_BASE = 990000").replace('f"/work/res_{iT:02d}.npz"', 'f"/work/res2_{iT:02d}.npz"')
assert src2 != src and "res2_" in src2 and "990000" in src2
open('/work/prod2.py','w').write(src2)
import subprocess, sys
r = subprocess.run([sys.executable, '/work/prod2.py', '0','1','2','3','4','5','6'], capture_output=True, text=True)
print(r.stdout); print(r.stderr[-3000:])
