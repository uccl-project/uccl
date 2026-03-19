// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PACKET_DEVICE_HPP_
#define MSCCLPP_PACKET_DEVICE_HPP_

#include "device.hpp"
#include <cstdint>
#include <type_traits>

#if defined(MSCCLPP_DEVICE_COMPILE)
#include "atomic_device.hpp"
#include "poll_device.hpp"
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

namespace mscclpp {
/// LL (low latency) protocol packet with 8 bytes of data and 8 bytes of flags.
union alignas(16) LL16Packet {
  // Assume data is written with an atomicity of 8 bytes (IB/RDMA).
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  using Payload = uint2;

#if defined(MSCCLPP_DEVICE_COMPILE)
  ulonglong2 raw_;

  MSCCLPP_INLINE LL16Packet() = default;

  MSCCLPP_DEVICE_INLINE LL16Packet(uint2 val, uint32_t flag)
      : data1(val.x), flag1(flag), data2(val.y), flag2(flag) {}

  /// Write 8 bytes of data to the packet.
  /// @param val1 The first 4-byte data to write.
  /// @param val2 The second 4-byte data to write.
  /// @param flag The flag to write.
  MSCCLPP_DEVICE_INLINE void write(uint32_t val1, uint32_t val2,
                                   uint32_t flag) {
#if defined(MSCCLPP_DEVICE_CUDA)
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(&raw_),
                 "r"(val1), "r"(flag), "r"(val2), "r"(flag));
#else  // !defined(MSCCLPP_DEVICE_CUDA)
    uint4 reg = make_uint4(val1, flag, val2, flag);
    ulonglong2* p = reinterpret_cast<ulonglong2*>(&reg);
    __builtin_nontemporal_store(p->x, &(raw_.x));
    __builtin_nontemporal_store(p->y, &(raw_.y));
#endif
  }

  /// Write 8 bytes of data to the packet.
  /// @param val The 8-byte data to write.
  /// @param flag The flag to write.
  MSCCLPP_DEVICE_INLINE void write(uint64_t val, uint32_t flag) {
    write((uint32_t)val, (uint32_t)(val >> 32), flag);
  }

  /// Write 8 bytes of data to the packet.
  /// @param val The 8-byte data to write.
  /// @param flag The flag to write.
  MSCCLPP_DEVICE_INLINE void write(uint2 val, uint32_t flag) {
    write(val.x, val.y, flag);
  }

  /// Helper of read().
  /// @param flag The flag to read.
  /// @param data The 8-byte data read.
  /// @return True if the flag is not equal to the given flag.
  MSCCLPP_DEVICE_INLINE bool readOnce(uint32_t flag, uint2& data) const {
#if defined(MSCCLPP_DEVICE_CUDA)
    uint32_t flag1, flag2;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(data.x), "=r"(flag1), "=r"(data.y), "=r"(flag2)
                 : "l"(&raw_));
    return (flag1 != flag) || (flag2 != flag);
#else  // !defined(MSCCLPP_DEVICE_CUDA)
    ulonglong2 reg;
    reg.x = atomicLoad(&(raw_.x), memoryOrderRelaxed);
    reg.y = atomicLoad(&(raw_.y), memoryOrderRelaxed);
    uint4* ptr = reinterpret_cast<uint4*>(&reg);
    data.x = ptr->x;
    data.y = ptr->z;
    return (ptr->y != flag) || (ptr->w != flag);
#endif
  }

  /// Read 8 bytes of data from the packet. It will spin until the flag is equal
  /// to the given flag.
  /// @param flag The flag to read.
  /// @param maxSpinCount The maximum number of spin counts before asserting.
  /// Never assert if negative.
  /// @return The 8-byte data read.
  MSCCLPP_DEVICE_INLINE uint2
  read(uint32_t flag, [[maybe_unused]] int64_t maxSpinCount = 100000000) const {
    uint2 data;
    POLL_MAYBE_JAILBREAK(readOnce(flag, data), maxSpinCount);
    return data;
  }

  /// Clear the packet.
  MSCCLPP_DEVICE_INLINE void clear() { raw_ = make_ulonglong2(0, 0); }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

/// LL (low latency) protocol packet with 4 bytes of data and 4 bytes of flags.
union alignas(8) LL8Packet {
  // Assume data is written with an atomicity of 8 bytes (IB/RDMA).
  struct {
    uint32_t data;
    uint32_t flag;
  };
  uint64_t raw_;

  using Payload = uint32_t;
#if defined(MSCCLPP_DEVICE_COMPILE)

  MSCCLPP_INLINE LL8Packet() = default;

  MSCCLPP_DEVICE_INLINE LL8Packet(uint32_t val, uint32_t flag)
      : data(val), flag(flag) {}

  /// Write 4 bytes of data to the packet.
  /// @param val The 4-byte data to write.
  /// @param flag The flag to write.
  MSCCLPP_DEVICE_INLINE void write(uint32_t val, uint32_t flag) {
#if defined(MSCCLPP_DEVICE_CUDA)
    asm volatile("st.volatile.global.v2.u32 [%0], {%1,%2};" ::"l"(&raw_),
                 "r"(val), "r"(flag));
#else  // !defined(MSCCLPP_DEVICE_CUDA)
    uint2 reg = make_uint2(val, flag);
    uint64_t* p = reinterpret_cast<uint64_t*>(&reg);
    atomicStore(&(raw_), *p, memoryOrderRelaxed);
#endif
  }

  /// Helper of read().
  /// @param flag The flag to read.
  /// @param data The 4-byte data read.
  /// @return True if the flag is not equal to the given flag.
  MSCCLPP_DEVICE_INLINE bool readOnce(uint32_t flag, uint32_t& data) const {
#if defined(MSCCLPP_DEVICE_CUDA)
    uint32_t f;
    asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];"
                 : "=r"(data), "=r"(f)
                 : "l"(&raw_));
    return (f != flag);
#else  // !defined(MSCCLPP_DEVICE_CUDA)
    uint64_t reg;
    reg = atomicLoad(&(raw_), memoryOrderRelaxed);
    uint2* ptr = reinterpret_cast<uint2*>(&reg);
    data = ptr->x;
    return (ptr->y != flag);
#endif
  }

  /// Read 4 bytes of data from the packet. It will spin until the flag is equal
  /// to the given flag.
  /// @param flag The flag to read.
  /// @param maxSpinCount The maximum number of spin counts before asserting.
  /// Never assert if negative.
  /// @return The 4-byte data read.
  MSCCLPP_DEVICE_INLINE uint32_t
  read(uint32_t flag, [[maybe_unused]] int64_t maxSpinCount = 1000000) const {
    uint32_t data;
    POLL_MAYBE_JAILBREAK(readOnce(flag, data), maxSpinCount);
    return data;
  }

  /// Clear the packet.
  MSCCLPP_DEVICE_INLINE void clear() { raw_ = 0; }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

using LLPacket = LL16Packet;

}  // namespace mscclpp

#endif  // MSCCLPP_PACKET_DEVICE_HPP_
