# NCCL Fast Socket

NCCL Fast Socket is a transport layer plugin to improve NCCL collective
communication performance on Google Cloud.

## Overview

Collective communication primitives such as all-reduce and all-gather have been
widely used in distributed training in machine learning. The NVIDIA Collective
Communications Library (NCCL) is a highly optimized implementation of these
multi-GPU and multi-node collective communication primitives that supports
NVIDIA GPUs.

NCCL Fast Socket is based on TCP/IP communication and uses a number of
techniques to achieve better and more consistent performance, especially with
100 Gbps networking on Google Cloud.

## Getting Started

### Dependencies

Fast Socket requires working installation of CUDA to build. After building the
plugin, it has to be in `LD_LIBRARY_PATH` in order to be loaded by NCCL.

### Build

The plugin uses Bazel to build. You can build the plugin as follows:

```
$ bazel build :all
```

The plugin is located at `bazel-bin/libnccl-net.so` and can be copied into your
`LD_LIBRARY_PATH`.

## Getting Help

Please open an issue if you have any questions or if you think you may have
found any bugs.

## Contributing

Contributions are always welcomed. Please refer to our [contributing guidelines](CONTRIBUTING.md)
to learn how to contriute.

## License

Fast Socket is licensed under the terms of a BSD-style license.
See [LICENSE](LICENSE) for more information.

This is not an officially supported Google product.
