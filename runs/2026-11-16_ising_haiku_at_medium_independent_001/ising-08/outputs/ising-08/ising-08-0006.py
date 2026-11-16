import subprocess
print(subprocess.run('ps aux | grep -c drive; ps aux --sort=-%cpu | head -15; nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null',shell=True,capture_output=True,text=True).stdout)