import subprocess, os
print(subprocess.run('ps aux | grep driver | head; ls -la /work; cat /work/logA.txt',shell=True,capture_output=True,text=True).stdout)
