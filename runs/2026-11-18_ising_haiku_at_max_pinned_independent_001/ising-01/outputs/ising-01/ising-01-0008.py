import subprocess
r = subprocess.run(["python3", "/work/prod.py", "5,6,7,8,9"], capture_output=True, text=True, timeout=590)
print(r.stdout)
print("STDERR:", r.stderr[-2000:])
