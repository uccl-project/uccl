import paramiko
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
    print(f'echo $$ ; exec /bin/bash -c "{command}"')
    stdin, stdout, stderr = sshclient.exec_command(
        f'echo $$ ; exec /bin/bash -c "{command}"'
    )
    pid = stdout.readline()
    cd = CommandDescriptor(sshclient, pid, stdout, stderr)
    return cd.wait()


def exec_command_no_wait(sshclient, command):
    stdin, stdout, stderr = sshclient.exec_command(
        f'echo $$ ; exec /bin/bash -c "{command}"'
    )
    pid = stdout.readline()
    return CommandDescriptor(sshclient, pid, stdout, stderr)


def get_nodes(path="nodes.txt"):
    with open(path, "r") as file:
        return [
            line.strip()
            for line in file
            if not line.strip().startswith("#") and line.strip()
        ]


def parse_num_queues(make_macro, file_path):
    after_marker = f"if defined({make_macro})"
    pattern = r"static\s+const\s+uint32_t\s+NUM_QUEUES\s*=\s*(\d+)\s*;"
    hit_marker = False
    with open(file_path, "r") as file:
        for line in file:
            if after_marker in line:
                hit_marker = True
            if hit_marker:
                match = re.search(pattern, line)
                if match:
                    return int(match.group(1))
    return None
