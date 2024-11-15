#include "transport_config.h"
#include "util.h"

using namespace uccl;

int main() {
    std::vector<uint32_t> redir_table;
    std::vector<uint8_t> rss_key;
    bool res = get_rss_config(DEV_DEFAULT, redir_table, rss_key);
    DCHECK(res);

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

    // Example UDP 4-tuple
    uint32_t src_ip = 0xC0A80001;  // 192.168.0.1
    uint32_t dst_ip = 0xC0A80002;  // 192.168.0.2
    uint16_t src_port = 12345;
    uint16_t dst_port = 80;

    // Calculate the queue ID
    uint32_t queue_id = calculate_queue_id(src_ip, dst_ip, src_port, dst_port,
                                           rss_key, redir_table);

    std::cout << "Queue ID: " << queue_id << std::endl;

    std::vector<uint16_t> dst_ports;
    res = get_dst_ports_with_target_queueid(
        src_ip, dst_ip, src_port, queue_id, rss_key, redir_table, 128, dst_ports);

    std::cout << "Destination ports: ";
    for (auto port : dst_ports) {
        std::cout << std::dec << port << " ";
    }
    std::cout << std::endl;

    return 0;
}