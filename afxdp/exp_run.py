import time
import re
import sys
sys.path.append("../")
from common import *

    
client_ip = "172.31.19.147"
server_ip = "172.31.20.99"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(client_ip, username="ubuntu")

server = paramiko.SSHClient()
server.set_missing_host_key_policy(paramiko.AutoAddPolicy())
server.connect(server_ip, username="ubuntu")


def run_afxdp_exp(inflight_pkt, payload_size):
    _ = exec_command_and_wait(
        server,
        f"cd uccl/afxdp; sed -i 's/const int MAX_INFLIGHT_PKTS = [0-9]\+;/const int MAX_INFLIGHT_PKTS = {inflight_pkt};/' client_main.cc",
    )
    _ = exec_command_and_wait(
        server,
        f"cd uccl/afxdp; sed -i 's/const int PAYLOAD_BYTES = [0-9]\+;/const int PAYLOAD_BYTES = {payload_size};/' client_main.cc",
    )

    _ = exec_command_and_wait(server, "cd uccl/afxdp; make clean; make -j")
    _ = exec_command_and_wait(server, f"cd uccl; ./sync.sh {client_ip}")
    _ = exec_command_and_wait(server, "cd uccl; ./config_nic.sh ens6 1 3498 afxdp")
    _ = exec_command_and_wait(client, "cd uccl; ./config_nic.sh ens6 1 3498 afxdp")

    server_desc = exec_command_no_wait(
        server, "cd uccl/afxdp; sudo ./server_main"
    )
    time.sleep(3)
    client_res = exec_command_and_wait(
        client, "cd uccl/afxdp; sudo ./client_main"
    )

    server_desc.kill()
    _ = server_desc.wait()

    return client_res[0]


def run_tcp_exp(inflight_pkt, payload_size):
    _ = exec_command_and_wait(
        server,
        f"cd uccl/afxdp; sed -i 's/const int MAX_INFLIGHT_PKTS = [0-9]\+;/const int MAX_INFLIGHT_PKTS = {inflight_pkt};/' client_tcp_main.cc",
    )
    _ = exec_command_and_wait(
        server,
        f"cd uccl/afxdp; sed -i 's/#define PAYLOAD_BYTES [0-9]\+/#define PAYLOAD_BYTES {payload_size}/' util_tcp.h",
    )

    _ = exec_command_and_wait(server, "cd uccl/afxdp; make clean; make -j")
    _ = exec_command_and_wait(server, f"cd uccl; ./sync.sh {client_ip}")
    _ = exec_command_and_wait(server, "cd uccl; ./config_nic.sh ens6 4 9001 tcp")
    _ = exec_command_and_wait(client, "cd uccl; ./config_nic.sh ens6 4 9001 tcp")

    server_desc = exec_command_no_wait(
        server, "cd uccl/afxdp; ./server_tcp_main"
    )
    time.sleep(3)
    client_res = exec_command_and_wait(
        client, "cd uccl/afxdp; ./client_tcp_main -a 172.31.22.249"
    )

    server_desc.kill()
    _ = server_desc.wait()

    return client_res[0]


def parse_results(log_str):
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
    tput = tput_value.group(1) if tput_value else "No throughput found"
    bw = bw_value.group(1) if bw_value else "No BW found"
    med_rtt = med_rtt_value.group(1) if med_rtt_value else "No med rtt found"
    tail_rtt = (
        tail_rtt_value.group(1) if tail_rtt_value else "No tail rtt found"
    )

    print(f"Throughput: {tput} Kpkts/s")
    print(f"BW: {bw} Gbps")
    print(f"med rtt: {med_rtt} us")
    print(f"tail rtt: {tail_rtt} us")
    return tput, bw, med_rtt, tail_rtt


legend_vec = ["afxdp", "tcp"]
inflight_pkt_vec = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
payload_size_vec = [64, 1024, 2048, 3072]


exp_res_file = open("exp_res.csv", "a")
exp_res_file.write(
    f"legend,inflight_pkt,payload_size,tput_kpkts,bw_gbps,med_rtt_us,tail_rtt_us\n"
)
exp_res_file.flush()

for legend in legend_vec:
    for inflight_pkt in inflight_pkt_vec:
        for payload_size in payload_size_vec:
            if legend == "afxdp":
                log_str = run_afxdp_exp(inflight_pkt, payload_size)
            elif legend == "tcp":
                log_str = run_tcp_exp(inflight_pkt, payload_size)

            # Getting benchmark results
            tput, bw, med_rtt, tail_rtt = parse_results(log_str)

            exp_res_file.write(
                f"{legend},{inflight_pkt},{payload_size},{tput},{bw},{med_rtt},{tail_rtt}\n"
            )
            exp_res_file.flush()

client.close()
server.close()
exp_res_file.close()
