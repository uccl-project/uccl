from pathlib import Path
p=Path('src/multinode/mscclpp_backend.cu');s=p.read_text()
s=s.replace('  uint64_t mediumSequence = 0;', '''  uint64_t mediumSequence = 0;
#ifdef UCCL_LITE_RS128_PREFIX_BANKS
  uint64_t prefix128Sequence = 0;
  bool prefix128Active = false;
  uint64_t prefix128Entries = 0, prefix128Reuses = 0, prefix128Leaves = 0;
#endif''',1)
old='''    Layout largeLayout = this->largeLayout;
    if (medium) {'''
new='''    Layout largeLayout = this->largeLayout;
    bool prefix128 = false;
#ifdef UCCL_LITE_RS128_PREFIX_BANKS
    constexpr size_t prefixCapacity = 128UL << 20;
    constexpr size_t prefixPage = 2UL << 20;
    constexpr size_t prefixSlot =
        (prefixCapacity / 4 + 4096 + prefixPage - 1) & ~(prefixPage - 1);
    constexpr size_t prefixStride = 3 * prefixCapacity + 2 * prefixSlot;
    // Two 452-MiB banks fit inside the already allocated large data region.
    // The full region is reused by larger algorithms, so synchronize entry
    // after any ordinary large-layout operation, not on every steady call.
    prefix128 = op == NetworkOperation::ReduceScatter &&
                bytes == prefixCapacity && capacity >= 2 * prefixStride;
    if (prefix128) {
      if (!prefix128Active) {
        ++prefix128Entries;
        reuseLarge<<<1, 32, 0, stream>>>(t.localDirect, t.ib, rank % 4);
        checked(cudaGetLastError());
      } else {
        ++prefix128Reuses;
      }
      prefix128Active = true;
      size_t offset = (prefix128Sequence++ & 1) * prefixStride;
      largeLayout = {offset, offset + prefixCapacity,
                     offset + 2 * prefixCapacity,
                     offset + 2 * prefixCapacity + prefixSlot,
                     offset + 2 * prefixCapacity + 2 * prefixSlot};
      medium = true;
    }
#endif
    if (medium && !prefix128) {'''
assert old in s;s=s.replace(old,new,1)
old='''      if (!medium) {
        reuseLarge<<<1, 32, 0, stream>>>'''
new='''      if (!medium) {
#ifdef UCCL_LITE_RS128_PREFIX_BANKS
        if (prefix128Active) ++prefix128Leaves;
        prefix128Active = false;
#endif
        reuseLarge<<<1, 32, 0, stream>>>'''
assert s.count(old)==2;s=s.replace(old,new)
s=s.replace('    if (pending) cudaEventSynchronize(done);', '    if (pending) cudaEventSynchronize(done);\n#ifdef UCCL_LITE_RS128_PREFIX_BANKS\n    std::fprintf(stderr,\n                 "UCCL Lite prefix128: rank=%d entries=%llu reuses=%llu leaves=%llu\\n",\n                 rank, (unsigned long long)prefix128Entries,\n                 (unsigned long long)prefix128Reuses,\n                 (unsigned long long)prefix128Leaves);\n#endif',1)
p.write_text(s)
