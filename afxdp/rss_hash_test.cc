#include "transport_config.h"
#include "util/rss.h"

using namespace uccl;

int main() {
  std::vector<uint8_t> default_rss_key = {
      0x6d, 0x5a, 0x56, 0xda, 0x25, 0x5b, 0x0e, 0xc2, 0x41, 0x67,
      0x25, 0x3d, 0x43, 0xa3, 0x8f, 0xb0, 0xd0, 0xca, 0x2b, 0xcb,
      0xae, 0x7b, 0x30, 0xb4, 0x77, 0xcb, 0x2d, 0xa3, 0x80, 0x30,
      0xf2, 0x0c, 0x6a, 0x42, 0xb7, 0x3b, 0xbe, 0xac, 0x01, 0xfa,
  };

  // Example UDP 4-tuple
  uint32_t dst_ip = 0xa18e6450;  // 161.142.100.80
  uint32_t src_ip = 0x420995bb;  // 66.9.149.187
  uint16_t dst_port = 1766;
  uint16_t src_port = 2794;

  std::vector<uint32_t> redir_table;
  std::vector<uint8_t> rss_key;
  bool res = get_rss_config(DEV_DEFAULT, redir_table, rss_key);
  DCHECK(res);

  uint32_t rss_hash =
      calculate_rss_hash(src_ip, dst_ip, src_port, dst_port, default_rss_key);
  CHECK_EQ(rss_hash, 0x51ccc178);

  calculate_queue_id(src_ip, dst_ip, src_port, dst_port, default_rss_key,
                     redir_table);

  std::cout << "RSS key: ";
  for (auto byte : rss_key) {
    std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)byte;
  }
  std::cout << std::endl;

  std::cout << "Redirection table: ";
  for (auto entry : redir_table) {
    std::cout << std::dec << entry << " ";
  }
  std::cout << std::endl;

  std::vector<uint8_t> rss_key_le;
  rss_key_le.resize(rss_key.size());
  rte_convert_rss_key((uint32_t const*)rss_key.data(),
                      (uint32_t*)rss_key_le.data(), rss_key.size());

  std::cout << "RSS key le: ";
  for (auto byte : rss_key_le) {
    std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)byte;
  }
  std::cout << std::endl;

  src_ip = 0xc0a80601;  // 192.168.6.1
  dst_ip = 0xc0a80602;  // 192.168.6.2
  src_port = BASE_PORT;
  dst_port = BASE_PORT;

  for (int i = BASE_PORT; i < BASE_PORT + 128; i++) {
    dst_port = i;
    src_port = src_port;
    dst_port = dst_port;
    uint32_t queue_id = calculate_queue_id(src_ip, dst_ip, src_port, dst_port,
                                           rss_key, redir_table);
    std::cout << std::dec << "dst_port " << i << " Queue ID " << queue_id
              << std::endl;
  }

  uint32_t queue_id = 0;

  std::vector<uint16_t> dst_ports;
  res =
      get_dst_ports_with_target_queueid(src_ip, dst_ip, src_port, queue_id,
                                        rss_key, redir_table, 1024, dst_ports);

  std::cout << "Destination ports: ";
  for (auto port : dst_ports) {
    std::cout << std::dec << port << " ";
  }
  std::cout << std::endl;

  return 0;
}