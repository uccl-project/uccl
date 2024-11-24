import paramiko


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


def read_nodes():
    with open("nodes.txt", "r") as file:
        return [
            line.strip()
            for line in file
            if not line.strip().startswith("#") and line.strip()
        ]
