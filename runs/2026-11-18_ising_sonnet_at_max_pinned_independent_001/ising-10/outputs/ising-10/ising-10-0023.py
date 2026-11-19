
import subprocess, sys
r = subprocess.run([sys.executable, '/work/prod3.py', '5','6','7','8','9'], capture_output=True, text=True)
print(r.stdout); print(r.stderr[-3000:])
