FROM quay.io/pypa/manylinux2014_x86_64

# 1. Install system dependencies
RUN yum install -y \
    glog-devel \
    gflags-devel \
    zlib-devel \
    elfutils-libelf-devel \
    libibverbs-devel \
    make \
    gcc-c++ \
    cmake \
    git \
    which

# 2. Set CUDA path (change to your CUDA version)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# If you have local CUDA, you can COPY instead (optional)
# COPY cuda /usr/local/cuda

# 3. Copy your entire repo into the container
# Expect repo root to be the context
WORKDIR /io
COPY . .

# 4. Build the shared library
RUN cd rdma && make -j$(nproc)

# 5. Move the .so into Python package directory for packaging
RUN mkdir -p uccl/lib && cp rdma/libnccl-net-uccl.so uccl/lib/

# 6. Build the wheel
RUN pip install build && python3 -m build

# 7. Repair the wheel for manylinux compliance
RUN pip install auditwheel && \
    auditwheel repair dist/uccl-*.whl -w /io/wheelhouse
