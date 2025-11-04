// epoll_server.h
// Header file containing epoll-based server implementation

#ifndef EPOLL_SERVER_H
#define EPOLL_SERVER_H

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// ---------------------------
// Serialization utilities
// ---------------------------
template <typename T>
std::string serialize(const T& obj) {
    std::string s(sizeof(T), '\0');
    std::memcpy(&s[0], reinterpret_cast<const char*>(&obj), sizeof(T));
    return s;
}

template <typename T>
T deserialize(const std::string& s) {
    T obj{};
    size_t copy_len = std::min(s.size(), sizeof(T));
    std::memcpy(reinterpret_cast<char*>(&obj), s.data(), copy_len);
    return obj;
}

// Message framing: [uint32_t len][payload bytes]

// ---------------------------
// Example MetaInfo struct
// ---------------------------
struct MetaInfo {
    int32_t client_id;
    double value;
    char message[64];
};

// ---------------------------
// Simple thread pool
// ---------------------------
class ThreadPool {
public:
    explicit ThreadPool(size_t nthreads) : stop_(false) {
        for (size_t i = 0; i < nthreads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mtx_);
                        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    try {
                        task();
                    } catch (const std::exception& e) {
                        std::cerr << "Task exception: " << e.what() << "\n";
                    }
                }
            });
        }
    }

    template <typename F>
    void enqueue(F&& f) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            tasks_.emplace(std::forward<F>(f));
        }
        cv_.notify_one();
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool stop_;
};

// ---------------------------
// Connection state
// ---------------------------
struct Connection {
    int fd = -1;
    std::vector<char> in_buf;   // accumulate incoming bytes
    std::vector<char> out_buf;  // pending bytes to send
    uint32_t expected_len = 0;  // 0 means expecting header (len)
    bool closed = false;
    sockaddr_in addr{};
};

// ---------------------------
// Helper functions
// ---------------------------

static int make_socket_non_blocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) return -1;
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) return -1;
    return 0;
}

// try send entire buffer, return bytes_sent
static ssize_t try_send(int fd, const char* buf, size_t len) {
    ssize_t n = ::send(fd, buf, len, MSG_NOSIGNAL);
    if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
        return -1;
    }
    return n;
}

// ---------------------------
// Epoll-based server class
// ---------------------------
class EpollServer {
public:
    using MetaHandler = std::function<void(const MetaInfo&, int client_fd)>;

    EpollServer(int port, MetaHandler handler, int thread_count = 4, int max_events = 1024)
        : port_(port),
          handler_(std::move(handler)),
          pool_(thread_count),
          max_events_(max_events),
          running_(false) {}

    ~EpollServer() { stop(); }

    bool start() {
        if (running_) return false;

        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
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
            ::close(listen_fd_);
            return false;
        }

        if (make_socket_non_blocking(listen_fd_) < 0) {
            perror("make nonblock");
            ::close(listen_fd_);
            return false;
        }

        if (listen(listen_fd_, SOMAXCONN) < 0) {
            perror("listen");
            ::close(listen_fd_);
            return false;
        }

        epoll_fd_ = epoll_create1(0);
        if (epoll_fd_ < 0) {
            perror("epoll_create1");
            ::close(listen_fd_);
            return false;
        }

        epoll_event ev{};
        ev.events = EPOLLIN | EPOLLET; // edge-triggered accept
        ev.data.fd = listen_fd_;
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, listen_fd_, &ev) < 0) {
            perror("epoll_ctl add listen");
            ::close(listen_fd_);
            ::close(epoll_fd_);
            return false;
        }

        running_ = true;
        worker_thread_ = std::thread([this] { this->event_loop(); });
        return true;
    }

    void stop() {
        if (!running_) return;
        running_ = false;
        // wake up epoll_wait by closing listen_fd_ or using eventfd; simplest: shutdown fd
        ::shutdown(listen_fd_, SHUT_RDWR);
        if (worker_thread_.joinable()) worker_thread_.join();
        ::close(listen_fd_);
        ::close(epoll_fd_);

        // close all client fds
        std::lock_guard<std::mutex> lk(conns_mtx_);
        for (auto& kv : conns_) {
            if (kv.second.fd >= 0) ::close(kv.second.fd);
        }
        conns_.clear();
    }

    // Run server in background (already start() does that). This function blocks if needed.
    void run_forever() {
        if (!start()) {
            std::cerr << "Failed to start server\n";
            return;
        }
        // block calling thread
        if (worker_thread_.joinable()) worker_thread_.join();
    }

private:
    void event_loop() {
        std::vector<epoll_event> events(max_events_);
        while (running_) {
            int n = epoll_wait(epoll_fd_, events.data(), max_events_, 1000);
            if (n < 0) {
                if (errno == EINTR) continue;
                perror("epoll_wait");
                break;
            }
            for (int i = 0; i < n; ++i) {
                epoll_event& ev = events[i];
                if ((ev.events & EPOLLERR) || (ev.events & EPOLLHUP)) {
                    // error on fd
                    int fd = ev.data.fd;
                    remove_connection(fd);
                    continue;
                }
                if (ev.data.fd == listen_fd_) {
                    // Accept all incoming connections (edge-triggered)
                    accept_loop();
                } else {
                    // Client socket events
                    if (ev.events & EPOLLIN) handle_read(ev.data.fd);
                    if (ev.events & EPOLLOUT) handle_write(ev.data.fd);
                }
            }
        }
    }

    void accept_loop() {
        while (true) {
            sockaddr_in in_addr;
            socklen_t in_len = sizeof(in_addr);
            int infd = accept(listen_fd_, (sockaddr*)&in_addr, &in_len);
            if (infd < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // all accepted
                    break;
                } else {
                    perror("accept");
                    break;
                }
            }
            if (make_socket_non_blocking(infd) < 0) {
                perror("make_socket_non_blocking(infd)");
                ::close(infd);
                continue;
            }
            // add to epoll
            epoll_event event{};
            event.data.fd = infd;
            event.events = EPOLLIN | EPOLLET | EPOLLOUT; // monitor both; EPOLLET for performance
            if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, infd, &event) < 0) {
                perror("epoll_ctl add client");
                ::close(infd);
                continue;
            }
            // create connection state
            Connection conn;
            conn.fd = infd;
            conn.addr = in_addr;
            conn.expected_len = 0;
            {
                std::lock_guard<std::mutex> lk(conns_mtx_);
                conns_.emplace(infd, std::move(conn));
            }
            char ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &in_addr.sin_addr, ip, sizeof(ip));
            std::cout << "Accepted connection fd=" << infd << " from " << ip << ":" << ntohs(in_addr.sin_port) << "\n";
        }
    }

    void handle_read(int fd) {
        // read as much as possible (edge-triggered)
        char buf[4096];
        while (true) {
            ssize_t count = ::recv(fd, buf, sizeof(buf), 0);
            if (count == -1) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) break; // all data read
                // error
                remove_connection(fd);
                return;
            } else if (count == 0) {
                // peer closed
                remove_connection(fd);
                return;
            } else {
                // got bytes
                std::lock_guard<std::mutex> lk(conns_mtx_);
                auto it = conns_.find(fd);
                if (it == conns_.end()) {
                    // connection not found
                    continue;
                }
                Connection& conn = it->second;
                conn.in_buf.insert(conn.in_buf.end(), buf, buf + count);
                // try to parse messages (maybe multiple)
                parse_messages(conn);
            }
        }
    }

    void parse_messages(Connection& conn) {
        // We expect frames of the form: uint32_t len (network byte order) + payload
        while (true) {
            if (conn.expected_len == 0) {
                // need header
                if (conn.in_buf.size() >= sizeof(uint32_t)) {
                    uint32_t net_len;
                    std::memcpy(&net_len, conn.in_buf.data(), sizeof(uint32_t));
                    uint32_t len = ntohl(net_len);
                    conn.expected_len = len;
                    // erase header
                    conn.in_buf.erase(conn.in_buf.begin(), conn.in_buf.begin() + sizeof(uint32_t));
                } else {
                    break; // wait for more
                }
            }
            if (conn.expected_len > 0) {
                if (conn.in_buf.size() >= conn.expected_len) {
                    // got full payload
                    std::string payload(conn.in_buf.begin(), conn.in_buf.begin() + conn.expected_len);
                    // consume payload
                    conn.in_buf.erase(conn.in_buf.begin(), conn.in_buf.begin() + conn.expected_len);
                    uint32_t got_len = conn.expected_len;
                    conn.expected_len = 0;

                    // Deserialize to MetaInfo (assuming size matches)
                    if (got_len != sizeof(MetaInfo)) {
                        std::cerr << "Warning: payload size mismatch. expected=" << sizeof(MetaInfo)
                                  << " got=" << got_len << "\n";
                    }
                    MetaInfo meta = deserialize<MetaInfo>(payload);

                    // Before handing to thread pool, we must send a response to client per your requirement.
                    std::string response = "Server ACK for client " + std::to_string(meta.client_id) + "\n";
                    // try to send immediately
                    ssize_t s = try_send(conn.fd, response.data(), response.size());
                    if (s < 0) {
                        // fatal send error -> close connection
                        std::cerr << "Error sending immediate response on fd=" << conn.fd << "\n";
                        remove_connection(conn.fd);
                        return;
                    } else if ((size_t)s < response.size()) {
                        // partial send -> buffer remainder and ensure EPOLLOUT monitored
                        conn.out_buf.insert(conn.out_buf.end(), response.data() + s, response.data() + response.size());
                        modify_epoll_out(conn.fd, true);
                    }
                    // Now dispatch to thread pool for processing
                    int client_fd = conn.fd;
                    MetaInfo meta_copy = meta;
                    pool_.enqueue([this, meta_copy, client_fd]() {
                        // user-provided handler
                        try {
                            handler_(meta_copy, client_fd);
                        } catch (...) {
                            std::cerr << "Handler exception\n";
                        }
                    });
                    // loop to see if more messages in buffer
                } else {
                    break; // wait for more payload bytes
                }
            }
        }
    }

    void handle_write(int fd) {
        std::lock_guard<std::mutex> lk(conns_mtx_);
        auto it = conns_.find(fd);
        if (it == conns_.end()) return;
        Connection& conn = it->second;
        while (!conn.out_buf.empty()) {
            ssize_t n = try_send(fd, conn.out_buf.data(), conn.out_buf.size());
            if (n < 0) {
                remove_connection(fd);
                return;
            } else if (n == 0) {
                // would block
                break;
            } else {
                conn.out_buf.erase(conn.out_buf.begin(), conn.out_buf.begin() + n);
            }
        }
        if (conn.out_buf.empty()) {
            // nothing left to send -> stop monitoring EPOLLOUT
            modify_epoll_out(fd, false);
        }
    }

    void modify_epoll_out(int fd, bool enable) {
        epoll_event ev{};
        ev.data.fd = fd;
        ev.events = EPOLLIN | EPOLLET;
        if (enable) ev.events |= EPOLLOUT;
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &ev) < 0) {
            // sometimes mod fails if fd is gone
            if (errno != ENOENT) perror("epoll_ctl mod");
        }
    }

    void remove_connection(int fd) {
        std::lock_guard<std::mutex> lk(conns_mtx_);
        auto it = conns_.find(fd);
        if (it == conns_.end()) return;
        ::close(it->second.fd);
        conns_.erase(it);
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr) < 0) {
            // ignore
        }
        std::cout << "Closed connection fd=" << fd << "\n";
    }

private:
    int port_;
    MetaHandler handler_;
    ThreadPool pool_;
    int max_events_;
    int listen_fd_ = -1;
    int epoll_fd_ = -1;
    std::thread worker_thread_;
    std::atomic<bool> running_;
    std::mutex conns_mtx_;
    std::map<int, Connection> conns_;
};

#endif // EPOLL_SERVER_H
