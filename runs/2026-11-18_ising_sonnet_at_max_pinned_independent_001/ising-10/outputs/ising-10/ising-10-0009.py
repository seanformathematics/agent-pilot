
import subprocess, sys
r = subprocess.run([sys.executable, '/work/prod.py', '7','8','9','10','11','12','13'], capture_output=True, text=True)
print(r.stdout); print(r.stderr[-3000:])
