FROM lmsysorg/sglang:latest
RUN apt-get update
RUN apt-get install -y linux-headers-generic
RUN apt -y install build-essential devscripts debhelper check libsubunit-dev fakeroot pkg-config dkms
RUN cd $HOME \
    && wget https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.4.tar.gz \
    && tar xf v2.4.tar.gz \
    && cd gdrcopy-2.4/packages \
    && CUDA=/usr/local/cuda ./build-deb-packages.sh \
    && dpkg -i gdrdrv-dkms_2.4-1_amd64.*.deb \
    && dpkg -i libgdrapi_2.4-1_amd64.*.deb \
    && dpkg -i gdrcopy-tests_2.4-1_amd64.*.deb \
    && dpkg -i gdrcopy_2.4-1_amd64.*.deb
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.43.2.tar.gz \
    && tar -xf aws-efa-installer-1.43.2.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y --skip-kmod -g --no-verify
RUN cd /opt \
    && git clone https://github.com/NVIDIA/nccl.git -b v2.23.4-1 \
    && cd nccl \
    && make -j src.build CUDA_HOME=/usr/local/cuda
RUN cd $HOME \
    && git clone https://github.com/NVIDIA/nccl-tests.git \
    && cd nccl-tests \
    && export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH \
    && make MPI=1 MPI_HOME=/opt/amazon/openmpi NCCL_HOME=/opt/nccl/build CUDA_HOME=/usr/local/cuda
WORKDIR /workspace/uccl/ep/deep_ep_wrapper/