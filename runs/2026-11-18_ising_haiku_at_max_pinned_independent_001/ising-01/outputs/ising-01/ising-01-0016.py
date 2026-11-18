prod3 = open("/work/prod2.py").read()
prod3 = prod3.replace("base_seed2 = 730000", "base_seed2 = 990000")
prod3 = prod3.replace('outfile = "/work/results_prod2.json"', 'outfile = "/work/results_prod3.json"')
prod3 = prod3.replace('f"/work/raw_prod2_T{T}.npz"', 'f"/work/raw_prod3_T{T}.npz"')
open("/work/prod3.py", "w").write(prod3)
import subprocess
r = subprocess.run(["python3", "/work/prod3.py", "0,1,2,3,4,5,6"], capture_output=True, text=True, timeout=590)
print(r.stdout)
print("STDERR:", r.stderr[-2000:])
