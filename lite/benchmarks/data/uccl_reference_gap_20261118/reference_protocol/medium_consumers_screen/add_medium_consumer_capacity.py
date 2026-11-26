from pathlib import Path
p=Path('src/multinode/async_host_gather.cuh');s=p.read_text()
s=s.replace('  int pipelineBlocks_ = 0;', '''  int pipelineBlocks_ = 0;
#ifdef UCCL_LITE_HOST_MEDIUM_CONSUMERS
  int pipelineMediumBlocks_ = 0;
#endif''',1)
needle='''    if (pipelineBlocks_ <=
        kHostPipelineControlBlocks + kHostPipelineUploadBlocks)'''
new='''#ifdef UCCL_LITE_HOST_MEDIUM_CONSUMERS
    static_assert(UCCL_LITE_HOST_MEDIUM_CONSUMERS > 0 &&
                  UCCL_LITE_HOST_MEDIUM_CONSUMERS <= 64);
    pipelineMediumBlocks_ =
        std::min(kHostPipelineUploadBlocks + UCCL_LITE_HOST_MEDIUM_CONSUMERS +
                     kHostPipelineControlBlocks,
                 resident * pipelineProperty.multiProcessorCount);
#endif
'''+needle
assert needle in s;s=s.replace(needle,new,1)
old='''    timelineEntries_ =
        1 + pipelineBlocks_ * 12 *'''
new='''    int timelineBlocks = pipelineBlocks_;
#ifdef UCCL_LITE_HOST_MEDIUM_CONSUMERS
    timelineBlocks = std::max(timelineBlocks, pipelineMediumBlocks_);
#endif
    timelineEntries_ =
        1 + timelineBlocks * 12 *'''
assert old in s;s=s.replace(old,new,1)
old='''      static_assert(UCCL_LITE_HOST_MEDIUM_CONSUMERS > 0);
      if (shardBytes <= (4UL << 20))
        launchBlocks =
            std::min(pipelineBlocks_, kHostPipelineControlBlocks +
                                          kHostPipelineUploadBlocks +
                                          UCCL_LITE_HOST_MEDIUM_CONSUMERS);'''
new='''      if (shardBytes <= (4UL << 20)) launchBlocks = pipelineMediumBlocks_;'''
assert old in s;s=s.replace(old,new,1);p.write_text(s)
import subprocess
subprocess.run(['build/format-tools/clang_format/data/bin/clang-format', '-i', str(p)], check=True)
