import time,subprocess
time.sleep(420)
print(open('/work/drive.log').read()); print(subprocess.run('ls /work',shell=True,capture_output=True,text=True).stdout)