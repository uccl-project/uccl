#!/bin/bash
# Run this script ON l41 to enable SSH from l40
# Usage: bash /home/yangz/nfs/zhongjie/uccl/experimental/lite/ep/setup_l41_ssh.sh

mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Add uccl-dev key
KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOtluFz7491tDMxrGJpg7ViEFyoRhCOMSOOyfdVo5h2M yangz@mibura-sky-test-01"
if ! grep -q "yangz@mibura-sky-test-01" ~/.ssh/authorized_keys 2>/dev/null; then
    echo "$KEY" >> ~/.ssh/authorized_keys
    echo "Added uccl-dev key"
fi

chmod 600 ~/.ssh/authorized_keys
echo "SSH setup complete. Test from l40: ssh l41 hostname"
