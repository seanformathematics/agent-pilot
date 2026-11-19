
src = open('/work/prod.py').read()
src3 = (src.replace("SEED_BASE = 880000", "SEED_BASE = 1110000")
           .replace('f"/work/res_{iT:02d}.npz"', 'f"/work/res3_{iT:02d}.npz"')
           .replace("nburn = 200", "nburn = 1500"))
assert "res3_" in src3 and "1110000" in src3 and "nburn = 1500" in src3
open('/work/prod3.py','w').write(src3)
import subprocess, sys
r = subprocess.run([sys.executable, '/work/prod3.py', '0','1','2','3','4'], capture_output=True, text=True)
print(r.stdout); print(r.stderr[-3000:])
