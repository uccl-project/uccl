#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <elf.h>
#include <link.h>
namespace uccl_lite::runtime {
inline std::string artifactIdentity(uintptr_t address) {
  struct Match {
    uintptr_t address;
    std::string id;
  } match{address, {}};
  dl_iterate_phdr(
      [](dl_phdr_info* info, size_t, void* data) {
        auto& match = *static_cast<Match*>(data);
        bool contains = false;
        for (int i = 0; i < info->dlpi_phnum; ++i) {
          auto const& p = info->dlpi_phdr[i];
          uintptr_t begin = info->dlpi_addr + p.p_vaddr;
          contains |= p.p_type == PT_LOAD && match.address >= begin &&
                      match.address - begin < p.p_memsz;
        }
        if (!contains) return 0;
        for (int i = 0; i < info->dlpi_phnum; ++i) {
          auto const& p = info->dlpi_phdr[i];
          if (p.p_type != PT_NOTE) continue;
          auto pos = reinterpret_cast<unsigned char const*>(info->dlpi_addr +
                                                            p.p_vaddr);
          auto end = pos + p.p_filesz;
          while (size_t(end - pos) >= sizeof(ElfW(Nhdr))) {
            auto const& note = *reinterpret_cast<ElfW(Nhdr) const*>(pos);
            pos += sizeof(note);
            size_t names = (size_t(note.n_namesz) + 3) & ~size_t(3);
            size_t bytes = (size_t(note.n_descsz) + 3) & ~size_t(3);
            if (names > size_t(end - pos)) break;
            auto name = pos;
            pos += names;
            if (bytes > size_t(end - pos)) break;
            if (note.n_type == NT_GNU_BUILD_ID && note.n_namesz == 4 &&
                !std::memcmp(name, "GNU", 4)) {
              char const* hex = "0123456789abcdef";
              for (size_t j = 0; j < note.n_descsz; ++j) {
                match.id += hex[pos[j] >> 4];
                match.id += hex[pos[j] & 15];
              }
              return 1;
            }
            pos += bytes;
          }
        }
        return 1;
      },
      &match);
  if (match.id.empty())
    throw std::runtime_error("probe artifact has no build identity");
  return match.id;
}
}  // namespace uccl_lite::runtime
