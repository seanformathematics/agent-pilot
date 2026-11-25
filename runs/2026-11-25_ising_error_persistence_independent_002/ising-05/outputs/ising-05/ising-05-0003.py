import subprocess
cmd = "cd /work && for T in 1.9 2.0 2.1 2.2 2.3 2.4; do python sim.py $T 1000 2000 20000 $((100+ $(echo $T*10 | bc))) ; done > /work/prod.log 2>&1"
# fall back to python-based seeds if bc missing
seeds={1.9:191,2.0:201,2.1:211,2.2:221,2.3:231,2.4:241}
cmd = "cd /work && " + " ; ".join(f"python sim.py {T} 1000 2000 20000 {s}" for T,s in seeds.items()) + " > /work/prod.log 2>&1"
p = subprocess.Popen(cmd, shell=True)
print("started", p.pid)
