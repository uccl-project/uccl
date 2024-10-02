import paramiko
import time
import re


class CommandDescriptor:
    def __init__(self, sshclient, pid, stdout, stderr):
        self.sshclient = sshclient
        self.pid = pid
        self.stdout = stdout
        self.stderr = stderr

    def wait(self):
        res = (self.stdout.read().decode(), self.stderr.read().decode())
        print(res[0], res[1])
        return res

    def kill(self):
        self.sshclient.exec_command(f"kill -s SIGINT {self.pid}")


def exec_command_and_wait(sshclient, command):
    # exec_command will return before the command is finished,
    # one must use read() to wait for the command to finish
    stdin, stdout, stderr = sshclient.exec_command(
        f"echo $$ ; exec /bin/bash -c '{command}'"
    )
    pid = stdout.readline()
    cd = CommandDescriptor(sshclient, pid, stdout, stderr)
    return cd.wait()


def exec_command_no_wait(sshclient, command):
    stdin, stdout, stderr = sshclient.exec_command(
        f"echo $$ ; exec /bin/bash -c '{command}'"
    )
    pid = stdout.readline()
    return CommandDescriptor(sshclient, pid, stdout, stderr)


client_ip = "172.31.19.147"
server_ip = "172.31.20.99"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(client_ip, username="ubuntu")

server = paramiko.SSHClient()
server.set_missing_host_key_policy(paramiko.AutoAddPolicy())
server.connect(server_ip, username="ubuntu")

_ = exec_command_and_wait(server, "cd uccl/afxdp; make clean; make -j")
_ = exec_command_and_wait(server, "cd uccl; ./sync.sh")
_ = exec_command_and_wait(server, "cd uccl; ./run.sh ens6 1")
_ = exec_command_and_wait(client, "cd uccl; ./run.sh ens6 1")

server_desc = exec_command_no_wait(server, "cd uccl/afxdp; sudo ./server")
time.sleep(3)
client_res = exec_command_and_wait(client, "cd uccl/afxdp; sudo ./client")

server_desc.kill()
server_res = server_desc.wait()

# Getting benchmark results
log_str = client_res[0]

# Regular expressions to extract tput, BW, med rtt, and tail rtt
tput_pattern = r"Throughput: ([\d\.]+) Kpkts/s"
bw_pattern = r"BW: ([\d\.]+) Gbps"
med_rtt_pattern = r"med rtt: (\d+) us"
tail_rtt_pattern = r"tail rtt: (\d+) us"

# Search for the values in the log string
tput_value = re.search(tput_pattern, log_str)
bw_value = re.search(bw_pattern, log_str)
med_rtt_value = re.search(med_rtt_pattern, log_str)
tail_rtt_value = re.search(tail_rtt_pattern, log_str)

# Extract and print the results
throughput = tput_value.group(1) if tput_value else "No throughput found"
bw = bw_value.group(1) if bw_value else "No BW found"
med_rtt = med_rtt_value.group(1) if med_rtt_value else "No med rtt found"
tail_rtt = tail_rtt_value.group(1) if tail_rtt_value else "No tail rtt found"

print(f"Throughput: {throughput} Kpkts/s")
print(f"BW: {bw} Gbps")
print(f"med rtt: {med_rtt} us")
print(f"tail rtt: {tail_rtt} us")

client.close()
server.close()
