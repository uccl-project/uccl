#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace uccl {

class Connection {
   private:
    uint64_t connection_id;  // unique connection ID
    std::vector<std::pair<uint16_t, uint16_t>> src_dst_ports;

    const uint16_t bootstrap_port = 40000;

   public:
    // This function bind this connection to a specific local network interface
    // with the IP specified by the interface. It also listens on incoming
    // Connect() requests to build a connection. Each connection is identified
    // by a unique connection_id, and uses multiple src+dst port combinations to
    // leverage multiple paths. Each connection exclusively occupies a NIC
    // queue. Under the hood, we leverage TCP to boostrap our connections.
    // Let's not consider multi-tenancy for now. 
    Connection(const char* interface_name, uint32_t nic_queue_id);
    ~Connection();

    // Connecting to a remote address.
    bool Connect(const char* remote_ip);

    // Sending the data by leveraging multiple port combinations.
    bool Send(const void* data, size_t len);

    // Receiving the data by leveraging multiple port combinations.
    bool Recv(void* data, size_t len);
};

}  // namespace uccl