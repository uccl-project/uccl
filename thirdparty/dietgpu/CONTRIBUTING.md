# Contributing to DietGPU

DietGPU is still in a fairly early stage, and it is being rapidly iterated on to support integration into networked collective communication frameworks for distributed ML computation.

The underlying rANS codec itself is fairly stable, but extensions to the library will include fused all-reduce kernel implementations, and specializations for more structured data (sparse data, dimensional correlations, etc). A CUB-like device library for fused kernel rANS usage is on the table as well.

Contributions are very welcome, but preferably once the library achieves some stability, as this is a very early release of the code.

## License

By contributing to DietGPU, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.
