import subprocess
r = subprocess.run(["python3", "/work/prod3.py", "12,13"], capture_output=True, text=True, timeout=590)
print(r.stdout)
print("STDERR:", r.stderr[-2000:])
