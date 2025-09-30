/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "rx_cmsg_parser.h"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace tcpx {
namespace rx {
namespace {

#ifndef SO_DEVMEM_HEADER
#define SO_DEVMEM_HEADER 98
#endif
#ifndef SO_DEVMEM_OFFSET
#define SO_DEVMEM_OFFSET 99
#endif
#ifndef SCM_DEVMEM_HEADER
#define SCM_DEVMEM_HEADER SO_DEVMEM_HEADER
#endif
#ifndef SCM_DEVMEM_OFFSET
#define SCM_DEVMEM_OFFSET SO_DEVMEM_OFFSET
#endif
#ifndef SCM_DEVMEM_DMABUF
#define SCM_DEVMEM_DMABUF 0x42
#endif

struct devmemvec {
  uint32_t frag_offset;
  uint32_t frag_size;
  uint32_t frag_token;
};

}  // namespace

CmsgParser::CmsgParser(const ParserConfig& config)
  : config_(config) {}

int CmsgParser::parse(const struct msghdr* msg, ScatterList& scatter_list) {
  if (!msg) return -EINVAL;

  scatter_list.clear();
  stats_.total_messages++;

  int user_buffer_count = 0;
  size_t host_buf_offset = 0;
  uint64_t dst_offset = 0;
  int num_cm = 0;

  struct msghdr* mutable_msg = const_cast<struct msghdr*>(msg);
  for (struct cmsghdr* cmsg = CMSG_FIRSTHDR(mutable_msg);
       cmsg != nullptr;
       cmsg = CMSG_NXTHDR(mutable_msg, cmsg)) {
    if (cmsg->cmsg_level != SOL_SOCKET ||
        (cmsg->cmsg_type != SCM_DEVMEM_OFFSET && cmsg->cmsg_type != SCM_DEVMEM_HEADER &&
         cmsg->cmsg_type != SCM_DEVMEM_DMABUF)) {
      continue;
    }

    num_cm++;
    const auto* dmv = reinterpret_cast<const devmemvec*>(CMSG_DATA(cmsg));
    if (!dmv) {
      stats_.parse_errors++;
      return -EINVAL;
    }

    const uint32_t frag_size = dmv->frag_size;
    if (cmsg->cmsg_type == SCM_DEVMEM_HEADER) {
      if (frag_size == 0) {
        continue;
      }

      if (!validateLinearFragment(frag_size, static_cast<uint32_t>(host_buf_offset))) {
        stats_.validation_errors++;
        return -EINVAL;
      }

      ScatterEntry entry{};
      entry.src_ptr = static_cast<char*>(config_.bounce_buffer) + host_buf_offset;
      entry.src_offset = static_cast<uint32_t>(host_buf_offset);
      entry.dst_offset = static_cast<uint32_t>(dst_offset);
      entry.length = frag_size;
      entry.is_devmem = false;
      entry.token = 0;
      entry.token_count = 0;
      scatter_list.entries.push_back(entry);
      scatter_list.linear_bytes += frag_size;
      scatter_list.total_bytes += frag_size;

      host_buf_offset += frag_size;
      dst_offset += frag_size;
      user_buffer_count++;
      stats_.linear_fragments++;
      continue;
    }

    if (!validateFragment(dmv->frag_offset, frag_size)) {
      stats_.validation_errors++;
      return -EINVAL;
    }

    ScatterEntry entry{};
    entry.src_ptr = static_cast<char*>(config_.dmabuf_base) + dmv->frag_offset;
    entry.src_offset = dmv->frag_offset;
    entry.dst_offset = static_cast<uint32_t>(dst_offset);
    entry.length = frag_size;
    entry.is_devmem = true;
    entry.token = dmv->frag_token;
    entry.token_count = 1;
    scatter_list.entries.push_back(entry);
    scatter_list.devmem_bytes += frag_size;
    scatter_list.total_bytes += frag_size;

    dst_offset += frag_size;
    stats_.devmem_fragments++;
  }

  if (num_cm == 0) {
    stats_.validation_errors++;
    return -ENOMSG;
  }
  if (dst_offset == 0) {
    stats_.validation_errors++;
    return -ENODATA;
  }

  if (user_buffer_count > 0 && !config_.bounce_buffer) {
    stats_.validation_errors++;
    return -EFAULT;
  }

  return 0;
}

bool CmsgParser::validateFragment(uint32_t offset, uint32_t length) const {
  if (!config_.dmabuf_base || config_.dmabuf_size == 0) {
    return true;  // nothing to validate against
  }
  if (offset + length > config_.dmabuf_size) {
    return false;
  }
  return true;
}

bool CmsgParser::validateLinearFragment(uint32_t size, uint32_t offset) const {
  if (!config_.bounce_buffer || config_.bounce_size == 0) {
    return false;
  }
  if (offset + size > config_.bounce_size) {
    return false;
  }
  return true;
}

bool CmsgParser::validate(const ScatterList& scatter_list,
                          size_t expected_total_bytes) const {
  if (scatter_list.total_bytes != expected_total_bytes) {
    return false;
  }
  if (!utils::validateScatterList(scatter_list, expected_total_bytes)) {
    return false;
  }
  return true;
}

namespace utils {

DevMemFragment extractDevMemFragment(const struct cmsghdr* cmsg) {
  DevMemFragment frag{};
  if (!cmsg) return frag;
  if (cmsg->cmsg_len >= CMSG_LEN(sizeof(devmemvec))) {
    const auto* data = reinterpret_cast<const devmemvec*>(CMSG_DATA(cmsg));
    frag.frag_offset = data->frag_offset;
    frag.frag_size = data->frag_size;
    frag.frag_token = data->frag_token;
  }
  return frag;
}

size_t calculateTotalSize(const ScatterList& scatter_list) {
  return scatter_list.total_bytes;
}

bool validateScatterList(const ScatterList& scatter_list, size_t expected_size) {
  if (scatter_list.entries.empty()) {
    return expected_size == 0;
  }

  auto entries = scatter_list.entries;
  std::sort(entries.begin(), entries.end(),
            [](const ScatterEntry& a, const ScatterEntry& b) {
              return a.dst_offset < b.dst_offset;
            });

  uint32_t expected_offset = 0;
  for (const auto& entry : entries) {
    if (entry.dst_offset != expected_offset) {
      return false;
    }
    expected_offset += entry.length;
  }

  return expected_offset == expected_size;
}

std::string dumpScatterList(const ScatterList& scatter_list) {
  std::ostringstream oss;
  oss << "ScatterList: " << scatter_list.entries.size() << " entries, "
      << scatter_list.total_bytes << " total bytes\n";
  oss << "  DevMem: " << scatter_list.devmem_bytes << " bytes\n";
  oss << "  Linear: " << scatter_list.linear_bytes << " bytes\n";

  for (size_t i = 0; i < scatter_list.entries.size(); ++i) {
    const auto& entry = scatter_list.entries[i];
    oss << "  [" << i << "] "
        << (entry.is_devmem ? "DEVMEM" : "LINEAR")
        << " src_off=" << entry.src_offset
        << " dst_off=" << entry.dst_offset
        << " len=" << entry.length;
    if (entry.is_devmem) {
      oss << " token=" << entry.token;
    }
    oss << "\n";
  }

  return oss.str();
}

}  // namespace utils
}  // namespace rx
}  // namespace tcpx








