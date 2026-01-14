FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 as dev-base

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ENV PYTHON_VER=3.10
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    /opt/conda/bin/conda install -y python=${PYTHON_VER} pytorch cmake ninja -c pytorch && \
    /opt/conda/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    /opt/conda/bin/conda clean -ya

ENV CMAKE_PREFIX_PATH="/opt/conda/bin/../:/opt/conda/lib/python3.9/site-packages/torch/share/cmake"

CMD bash
