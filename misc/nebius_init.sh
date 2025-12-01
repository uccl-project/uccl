# cloud-config to bootstrap a Nebius instance. This should take about 15 minutes to complete with 40GB of disk space. 

users:
 - name: ubuntu
   sudo: ALL=(ALL) NOPASSWD:ALL
   shell: /bin/bash
   ssh_authorized_keys:
    - ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHqAvZn3ItK26vfLo3ZR03j6eAVa4hRZyOg2YxXr4Rqk uccl-dev

package_update: true

runcmd:
  # Base packages (system-level, as root)
  - [bash, -lc, "DEBIAN_FRONTEND=noninteractive apt-get update"]
  - [bash, -lc, "DEBIAN_FRONTEND=noninteractive apt-get install -y git linux-tools-$(uname -r) linux-headers-$(uname -r) dkms clang llvm cmake m4 build-essential net-tools libgoogle-glog-dev libgtest-dev libgflags-dev libelf-dev libpcap-dev libc6-dev-i386 libpci-dev libopenmpi-dev libibverbs-dev clang-format wget ca-certificates"]

  # Install Miniconda (user-local)
  - [sudo, "-u", "ubuntu", bash, -lc, "cd /home/ubuntu && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh"]
  - [sudo, "-u", "ubuntu", bash, -lc, "cd /home/ubuntu && bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/ubuntu/miniconda3"]
  - [sudo, "-u", "ubuntu", bash, -lc, "printf '%s\n' 'export PATH=$HOME/miniconda3/bin:$PATH' >> /home/ubuntu/.bashrc"]
  - [sudo, "-u", "ubuntu", bash, -lc, "export PATH=$HOME/miniconda3/bin:$PATH && $HOME/miniconda3/bin/conda init bash"]
  - [sudo, "-u", "ubuntu", bash, -lc, "export PATH=$HOME/miniconda3/bin:$PATH && pip install --upgrade pip && pip install paramiko pybind11"]
  - [sudo, "-u", "ubuntu", bash, -lc, "export PATH=$HOME/miniconda3/bin:$PATH && conda install -y -c conda-forge 'libstdcxx-ng>=12' 'libgcc-ng>=12'"]
  - [sudo, "-u", "ubuntu", bash, -lc, "export PATH=$HOME/miniconda3/bin:$PATH && pip uninstall nvidia-nvshmem-cu12 -y"]
  - [sudo, "-u", "ubuntu", bash, -lc, "rm -r $HOME/miniconda3/lib/python3.13/site-packages/nvidia/nvshmem/"]

  # Install NVSHMEM from CUDA apt repo
  - [bash, -lc, "cd /tmp && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"]
  - [bash, -lc, "cd /tmp && dpkg -i cuda-keyring_1.1-1_all.deb"]
  - [bash, -lc, "DEBIAN_FRONTEND=noninteractive apt-get update"]
  - [bash, -lc, "DEBIAN_FRONTEND=noninteractive apt-get -y install nvshmem nvshmem-cuda-12"]
  - [sudo, "-u", "ubuntu", bash, -lc, "printf '%s\n' 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvshmem/12:$LD_LIBRARY_PATH' >> /home/ubuntu/.bashrc"]
  - [sudo, "-u", "ubuntu", bash, -lc, "printf '%s\n' 'export PATH=/usr/bin/nvshmem_12:$PATH' >> /home/ubuntu/.bashrc"]
  - [bash, -lc, "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvshmem/12:$LD_LIBRARY_PATH && export PATH=/usr/bin/nvshmem_12:$PATH && nvshmem-info -a || echo 'WARNING: nvshmem error'"]

  # Build and install UCCL-EP
  - [sudo, "-u", "ubuntu", bash, -lc, "cd /home/ubuntu && git clone https://github.com/uccl-project/uccl.git --recursive"]
  - [sudo, "-u", "ubuntu", bash, -lc, "printf '%s\n' 'export UCCL_HOME=$HOME/uccl' >> /home/ubuntu/.bashrc"]
  - [sudo, "-u", "ubuntu", bash, -lc, "export PATH=$HOME/miniconda3/bin:$PATH && cd $HOME/uccl/ep && ./install_deps.sh && make -j install"]
  - [sudo, "-u", "ubuntu", bash, -lc, "export PATH=$HOME/miniconda3/bin:$PATH && python -c 'import torch, uccl.ep' || echo 'WARNING: torch/uccl.ep import test failed'"]
  - [sudo, "-u", "ubuntu", bash, -lc, "export PATH=$HOME/miniconda3/bin:$PATH && cd $HOME/uccl/thirdparty/DeepEP && python setup.py install"]
  - [sudo, "-u", "ubuntu", bash, -lc, "export PATH=$HOME/miniconda3/bin:$PATH && python -c 'import deep_ep' || echo 'WARNING: deep_ep import test failed'"]

  # Install GDRCopy debs (system-wide, as root)
  - [bash, -lc, "cd /tmp && wget https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/ubuntu24_04/x64/gdrcopy-tests_2.5.1-1_amd64.Ubuntu24_04+cuda12.8.deb"]
  - [bash, -lc, "cd /tmp && wget https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/ubuntu24_04/x64/gdrcopy_2.5.1-1_amd64.Ubuntu24_04.deb"]
  - [bash, -lc, "cd /tmp && wget https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/ubuntu24_04/x64/gdrdrv-dkms_2.5.1-1_amd64.Ubuntu24_04.deb"]
  - [bash, -lc, "cd /tmp && wget https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/ubuntu24_04/x64/libgdrapi_2.5.1-1_amd64.Ubuntu24_04.deb"]
  - [bash, -lc, "cd /tmp && DEBIAN_FRONTEND=noninteractive apt-get install -y ./gdrdrv-dkms_2.5.1-1_amd64.Ubuntu24_04.deb ./libgdrapi_2.5.1-1_amd64.Ubuntu24_04.deb ./gdrcopy_2.5.1-1_amd64.Ubuntu24_04.deb ./gdrcopy-tests_2.5.1-1_amd64.Ubuntu24_04+cuda12.8.deb || echo 'WARNING: GDRCopy install had issues'"]

  # Enable native IBGDA: NVIDIA module options
  - [bash, -lc, "echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords=\"PeerMappingOverride=1;\"' > /etc/modprobe.d/nvidia.conf"]
  - [bash, -lc, "update-initramfs -u"]
  - [bash, -lc, "reboot"]
