#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace uccl {

typedef uint64_t ConnectionID;

class Endpoint {
   private:
    struct ConnectionState {
        std::vector<uint16_t> src_ports;
    };

    std::unordered_map<ConnectionID, std::unique_ptr<ConnectionState>>
        state_map;

    const uint16_t bootstrap_port = 40000;

   public:
    // This function bind this endpoint to a specific local network interface
    // with the IP specified by the interface. It also listens on incoming
    // Connect() requests to estabish connections. Each connection is identified
    // by a unique connection_id, and uses multiple src+dst port combinations to
    // leverage multiple paths. Under the hood, we leverage TCP to boostrap our
    // connections. We do not consider multi-tenancy for now, assuming this
    // endpoint exclusively uses the NIC and its all queues.
    Endpoint(const char* interface_name);
    ~Endpoint();

    // Connecting to a remote address.
    ConnectionID Connect(const char* remote_ip);

    // Sending the data by leveraging multiple port combinations.
    bool Send(ConnectionID connection_id, const void* data, size_t len);

    // Receiving the data by leveraging multiple port combinations.
    bool Recv(ConnectionID connection_id, void* data, size_t len);
};

}  // namespace uccl