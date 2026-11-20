
import subprocess
print(subprocess.run('ps aux | grep run.py | grep -v grep; uptime',shell=True,capture_output=True,text=True).stdout)
