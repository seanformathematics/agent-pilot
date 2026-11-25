import subprocess,os,time
print(subprocess.run("ps aux | grep sim.py | head; ls -la /work; cat /work/prod.log",shell=True,capture_output=True,text=True).stdout)
