import time,subprocess
time.sleep(580)
print(open('/work/drive.log').read()); print(subprocess.run('ls /work; uptime',shell=True,capture_output=True,text=True).stdout)