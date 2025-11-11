// metadata_exchange.h
// Simple TCP-based metadata exchange for RDMA

#ifndef METADATA_EXCHANGE_H
#define METADATA_EXCHANGE_H

#include "define.h"
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <condition_variable>
#include <memory>
#include <atomic>
#include <functional>
#include <thread>
#include <mutex>
#include <cstring>
#include <iostream>

// Helper functions for socket operations
namespace metadata_exchange_utils {

// Send metadata over socket
inline bool send_metadata(int sockfd, const metadata& meta) {
    ssize_t sent = send(sockfd, &meta, sizeof(metadata), 0);
    if (sent != sizeof(metadata)) {
        perror("send_metadata");
        return false;
    }
    return true;
}

// Receive metadata from socket
inline bool recv_metadata(int sockfd, metadata* meta) {
    ssize_t received = recv(sockfd, meta, sizeof(metadata), MSG_WAITALL);
    if (received != sizeof(metadata)) {
        perror("recv_metadata");
        return false;
    }
    return true;
}

} // namespace metadata_exchange_utils

// Simple metadata server for receiving metadata from peers
class MetadataServer {
public:
    using MetadataHandler = std::function<void(const metadata&, int client_fd)>;

    MetadataServer(int port, MetadataHandler handler)
        : port_(port), handler_(std::move(handler)), running_(false), listen_fd_(-1) {}

    ~MetadataServer() { stop(); }

    bool start() {
        if (running_) return false;

        listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) {
            perror("socket");
            return false;
        }

        int opt = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port_);

        if (bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("bind");
            close(listen_fd_);
            return false;
        }

        if (listen(listen_fd_, SOMAXCONN) < 0) {
            perror("listen");
            close(listen_fd_);
            return false;
        }

        running_ = true;
        server_thread_ = std::thread([this] { this->server_loop(); });

        std::cout << "MetadataServer started on port " << port_ << "\n";
        return true;
    }

    void stop() {
        if (!running_) return;
        running_ = false;
        if (listen_fd_ >= 0) {
            shutdown(listen_fd_, SHUT_RDWR);
            close(listen_fd_);
            listen_fd_ = -1;
        }
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
    }

private:
    void server_loop() {
        while (running_) {
            sockaddr_in client_addr{};
            socklen_t addr_len = sizeof(client_addr);
            int client_fd = accept(listen_fd_, (sockaddr*)&client_addr, &addr_len);

            if (client_fd < 0) {
                if (errno == EINVAL || errno == EBADF) break; // Server stopped
                perror("accept");
                continue;
            }

            // Handle client in a new thread
            std::thread([this, client_fd, client_addr]() {
                this->handle_client(client_fd, client_addr);
            }).detach();
        }
    }

    void handle_client(int client_fd, sockaddr_in client_addr) {
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, ip, sizeof(ip));
        std::cout << "MetadataServer: Accepted connection from "
                  << ip << ":" << ntohs(client_addr.sin_port) << "\n";

        // Receive metadata
        metadata meta;
        if (!metadata_exchange_utils::recv_metadata(client_fd, &meta)) {
            std::cerr << "Failed to receive metadata from " << ip << "\n";
            close(client_fd);
            return;
        }

        std::cout << "MetadataServer: Received metadata from " << ip
                  << " (qpn=" << meta.qpn << ")\n";

        // Call handler
        if (handler_) {
            try {
                handler_(meta, client_fd);
            } catch (...) {
                std::cerr << "Handler exception\n";
            }
        }

        // Send ACK
        const char* ack = "OK";
        send(client_fd, ack, strlen(ack), 0);

        close(client_fd);
    }

    int port_;
    MetadataHandler handler_;
    std::atomic<bool> running_;
    int listen_fd_;
    std::thread server_thread_;
};

// Simple metadata client for sending metadata to peers
class MetadataClient {
public:
    MetadataClient() = default;

    // Send metadata to server and get response
    bool send_metadata_sync(const std::string& server_ip, int server_port,
                           const metadata& meta, std::string* response = nullptr,
                           int timeout_ms = 5000) {
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            perror("socket");
            return false;
        }

        // Set timeout
        struct timeval tv;
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        sockaddr_in server_addr{};
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(server_port);
        if (inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr) <= 0) {
            std::cerr << "Invalid address: " << server_ip << "\n";
            close(sockfd);
            return false;
        }

        if (connect(sockfd, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            perror("connect");
            close(sockfd);
            return false;
        }

        std::cout << "MetadataClient: Connected to " << server_ip << ":" << server_port << "\n";

        // Send metadata
        if (!metadata_exchange_utils::send_metadata(sockfd, meta)) {
            std::cerr << "Failed to send metadata\n";
            close(sockfd);
            return false;
        }

        std::cout << "MetadataClient: Sent metadata (qpn=" << meta.qpn << ")\n";

        // Receive response
        if (response) {
            char buf[256];
            ssize_t n = recv(sockfd, buf, sizeof(buf) - 1, 0);
            if (n > 0) {
                buf[n] = '\0';
                *response = buf;
                std::cout << "MetadataClient: Received response: " << *response << "\n";
            }
        }

        close(sockfd);
        return true;
    }
};

// Helper class for two-node metadata exchange
class MetadataExchanger {
public:
    MetadataExchanger(int my_rank, int port_base = 12345)
        : my_rank_(my_rank), port_base_(port_base), received_(false) {}

    // Initialize server
    bool init() {
        // Start server with handler
        auto handler = [this](const metadata& meta, int) {
            std::lock_guard<std::mutex> lock(mtx_);
            remote_meta_ = meta;
            received_ = true;
            cv_.notify_all();
        };

        server_ = std::make_unique<MetadataServer>(port_base_ + my_rank_, handler);
        if (!server_->start()) {
            std::cerr << "Failed to start metadata server on port "
                      << (port_base_ + my_rank_) << "\n";
            return false;
        }

        std::cout << "MetadataExchanger initialized for rank " << my_rank_
                  << " on port " << (port_base_ + my_rank_) << "\n";
        return true;
    }

    // Exchange metadata with peer
    bool exchange(const std::string& peer_ip, int peer_rank, const metadata& my_meta,
                  metadata* peer_meta, int timeout_ms = 5000) {
        // Send my metadata to peer
        int peer_port = port_base_ + peer_rank;

        MetadataClient client;
        std::string response;
        if (!client.send_metadata_sync(peer_ip, peer_port, my_meta, &response, timeout_ms)) {
            std::cerr << "Failed to send metadata to peer rank " << peer_rank << "\n";
            return false;
        }

        std::cout << "Sent metadata to " << peer_ip << ":" << peer_port << "\n";

        // Wait for peer's metadata
        std::unique_lock<std::mutex> lock(mtx_);
        if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                         [this] { return received_; })) {
            std::cerr << "Timeout waiting for peer metadata from rank " << peer_rank << "\n";
            return false;
        }

        if (peer_meta) {
            *peer_meta = remote_meta_;
        }

        std::cout << "Received metadata from peer rank " << peer_rank << "\n";
        return true;
    }

    void stop() {
        if (server_) server_->stop();
    }

private:
    int my_rank_;
    int port_base_;
    std::unique_ptr<MetadataServer> server_;

    std::mutex mtx_;
    std::condition_variable cv_;
    metadata remote_meta_;
    bool received_;
};

#endif // METADATA_EXCHANGE_H
