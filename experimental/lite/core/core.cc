// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "core.hpp"
#include "api.h"
#include <sstream>

namespace mscclpp {

MSCCLPP_API_CPP std::string version() {
  std::stringstream ss;
  ss << MSCCLPP_MAJOR << "." << MSCCLPP_MINOR << "." << MSCCLPP_PATCH;
  return ss.str();
}

MSCCLPP_API_CPP TransportFlags::TransportFlags(Transport transport)
    : detail::TransportFlagsBase(1 << static_cast<size_t>(transport)) {}

MSCCLPP_API_CPP bool TransportFlags::has(Transport transport) const {
  return detail::TransportFlagsBase::test(static_cast<size_t>(transport));
}

MSCCLPP_API_CPP bool TransportFlags::none() const {
  return detail::TransportFlagsBase::none();
}

MSCCLPP_API_CPP bool TransportFlags::any() const {
  return detail::TransportFlagsBase::any();
}

MSCCLPP_API_CPP bool TransportFlags::all() const {
  return detail::TransportFlagsBase::all();
}

MSCCLPP_API_CPP size_t TransportFlags::count() const {
  return detail::TransportFlagsBase::count();
}

MSCCLPP_API_CPP TransportFlags& TransportFlags::operator|=(
    TransportFlags other) {
  detail::TransportFlagsBase::operator|=(other);
  return *this;
}

MSCCLPP_API_CPP TransportFlags
TransportFlags::operator|(TransportFlags other) const {
  return TransportFlags(*this) |= other;
}

MSCCLPP_API_CPP TransportFlags
TransportFlags::operator|(Transport transport) const {
  return *this | TransportFlags(transport);
}

MSCCLPP_API_CPP TransportFlags& TransportFlags::operator&=(
    TransportFlags other) {
  detail::TransportFlagsBase::operator&=(other);
  return *this;
}

MSCCLPP_API_CPP TransportFlags
TransportFlags::operator&(TransportFlags other) const {
  return TransportFlags(*this) &= other;
}

MSCCLPP_API_CPP TransportFlags
TransportFlags::operator&(Transport transport) const {
  return *this & TransportFlags(transport);
}

MSCCLPP_API_CPP TransportFlags& TransportFlags::operator^=(
    TransportFlags other) {
  detail::TransportFlagsBase::operator^=(other);
  return *this;
}

MSCCLPP_API_CPP TransportFlags
TransportFlags::operator^(TransportFlags other) const {
  return TransportFlags(*this) ^= other;
}

MSCCLPP_API_CPP TransportFlags
TransportFlags::operator^(Transport transport) const {
  return *this ^ TransportFlags(transport);
}

MSCCLPP_API_CPP TransportFlags TransportFlags::operator~() const {
  return TransportFlags(*this).flip();
}

MSCCLPP_API_CPP bool TransportFlags::operator==(TransportFlags other) const {
  return detail::TransportFlagsBase::operator==(other);
}

MSCCLPP_API_CPP bool TransportFlags::operator!=(TransportFlags other) const {
  return detail::TransportFlagsBase::operator!=(other);
}

MSCCLPP_API_CPP detail::TransportFlagsBase TransportFlags::toBitset() const {
  return *this;
}

TransportFlags::TransportFlags(detail::TransportFlagsBase bitset)
    : detail::TransportFlagsBase(bitset) {}

const TransportFlags NoTransports = TransportFlags();

const TransportFlags AllIBTransports =
    Transport::IB0 | Transport::IB1 | Transport::IB2 | Transport::IB3 |
    Transport::IB4 | Transport::IB5 | Transport::IB6 | Transport::IB7;

const TransportFlags AllTransports =
    AllIBTransports | Transport::CudaIpc | Transport::Ethernet;

std::ostream& operator<<(std::ostream& os, Transport const& transport) {
  static const std::string TransportNames[] = {"UNK", "IPC", "IB0", "IB1",
                                               "IB2", "IB3", "IB4", "IB5",
                                               "IB6", "IB7", "ETH", "NUM"};
  os << TransportNames[static_cast<size_t>(transport)];
  return os;
}

std::ostream& operator<<(std::ostream& os, DeviceType const& deviceType) {
  static const std::string DeviceTypeNames[] = {"Unknown", "CPU", "GPU"};
  os << DeviceTypeNames[static_cast<size_t>(deviceType)];
  return os;
}

std::ostream& operator<<(std::ostream& os, Device const& device) {
  os << "Device(type=" << device.type << ", id=" << device.id << ")";
  return os;
}

}  // namespace mscclpp
