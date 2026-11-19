from pathlib import Path
p=Path('src/multinode/device_collectives.cuh');s=p.read_text();s=s.replace('#if defined(UCCL_LITE_EARLY_PACKET_FLUSH) || defined(UCCL_LITE_AR_EARLY_FLUSH)\n__device__', '#if defined(UCCL_LITE_EARLY_PACKET_FLUSH) || defined(UCCL_LITE_AR_EARLY_FLUSH) || defined(UCCL_LITE_AG_DIRECT128)\n__device__',1)
a=s.index('template <bool ClientIO = false, bool Packet = false>\n__device__ __forceinline__ void hierarchicalAllGatherSerialBody(');b=s.index('\ntemplate <bool ClientIO = false>\n__global__ void hierarchicalAllGatherSerial(',a);t=s[a:b]
old='''#ifdef UCCL_LITE_EARLY_PACKET_FLUSH
  uint64_t flushTicket = 0;
#endif
  if constexpr (Packet) {'''
new='''  uint64_t flushTicket = 0;
  bool directRetirement = false;
#ifdef UCCL_LITE_AG_DIRECT128
  if constexpr (ClientIO && !Packet) directRetirement = nelem == 32768;
#endif
  if constexpr (Packet) {'''
assert old in t;t=t.replace(old,new,1)
old='''  } else if (threadIdx.x == 0)
    ib.putWithSignal(layout.data + ownOff * sizeof(float), layout.ibSend,
                     shard * sizeof(float));'''
new='''  } else if (threadIdx.x == 0) {
#ifdef UCCL_LITE_AG_DIRECT128
    if (directRetirement)
      // Raw data plus the existing MSCCL++ receive signal and send-retirement
      // request. This is the only IB write; local fan-out cannot alter ibSend.
      flushTicket = ib.fifo_.push(
          {mscclpp::TriggerData | mscclpp::TriggerFlag | mscclpp::TriggerSync,
           ib.dst_, layout.data + ownOff * sizeof(float), ib.src_, layout.ibSend,
           shard * sizeof(float), ib.semaphoreId_});
    else
#endif
      ib.putWithSignal(layout.data + ownOff * sizeof(float), layout.ibSend,
                       shard * sizeof(float));
  }'''
assert old in t;t=t.replace(old,new,1)
t=t.replace('''      } else
        ib.flush();''','''      } else if (!directRetirement)
        ib.flush();''',1)
t=t.replace('''#else
      ib.flush();
#endif
    }
  } else if (threadIdx.x == 0)''','''#else
      if (!directRetirement) ib.flush();
#endif
    }
  } else if (threadIdx.x == 0)''',1)
needle='''    UCCL_LITE_SERIAL_END(7);
  }
#undef'''
replace='''#ifdef UCCL_LITE_AG_DIRECT128
    if (directRetirement && threadIdx.x == 0) endPacketFlush(ib, flushTicket);
#endif
    UCCL_LITE_SERIAL_END(7);
  }
#undef'''
assert needle in t;t=t.replace(needle,replace,1)
s=s[:a]+t+s[b:];p.write_text(s)
p=Path('src/multinode/mscclpp_backend.cu');s=p.read_text()
old='''    if (bytes <= (32UL << 10) ||
        (op == NetworkOperation::AllReduce && bytes <= (64UL << 10)) ||
        (op == NetworkOperation::AllGather && bytes <= (128UL << 10))) {'''
new='''    bool directGather128 = false;
#ifdef UCCL_LITE_AG_DIRECT128
    directGather128 = op == NetworkOperation::AllGather && bytes == (128UL << 10);
#endif
    if (!directGather128 &&
        (bytes <= (32UL << 10) ||
         (op == NetworkOperation::AllReduce && bytes <= (64UL << 10)) ||
         (op == NetworkOperation::AllGather && bytes <= (128UL << 10)))) {'''
assert old in s;s=s.replace(old,new,1);p.write_text(s)
