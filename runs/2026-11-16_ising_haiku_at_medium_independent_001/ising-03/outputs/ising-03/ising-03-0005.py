
import subprocess
print(subprocess.run('ps aux | grep prod | head; ls -la /work; cat /work/prod.log',shell=True,capture_output=True,text=True).stdout)
