#pragma once
#include "define.h"
#include "rdma_device.h"
#include "rdma_context.h"
#include "efa_channel.h"
#include "epoll_client.h"
#include "epoll_server.h"

// ChannelGroup manages multiple channels for a connection
class ChannelGroup {
public:
    ChannelGroup() = default;

    void addChannel(std::shared_ptr<EFAChannel> channel) {
        channels_.push_back(channel);
    }

    std::shared_ptr<EFAChannel> getChannel(size_t index) const {
        if (index >= channels_.size()) return nullptr;
        return channels_[index];
    }

    size_t channelCount() const {
        return channels_.size();
    }

    const std::vector<std::shared_ptr<EFAChannel>>& channels() const {
        return channels_;
    }

private:
    std::vector<std::shared_ptr<EFAChannel>> channels_;
    std::shared_ptr<EFAChannel> ctrl_channel;
};

// EFAEndpoint manages RDMA contexts and channel groups
class EFAEndpoint {
public:
    // Constructor with required gpu_index and optional device_ids
    // If device_ids is empty, all available devices will be used
    explicit EFAEndpoint(int gpu_index, uint64_t rank_id,
                        uint64_t port = 0,
                        const std::vector<size_t>& device_ids = std::vector<size_t>())
        : gpu_index_(gpu_index) ,rank_id_(rank_id), port_(port){
        
        initializeContexts(device_ids);
        std::cout << "EFAEndpoint initialized with " << contexts_.size()
                 << " context(s) for GPU " << gpu_index_ << std::endl;

        oob_server_ = std::make_shared<EpollServer>(port_, [this](const std::string& input, std::string& output) {
                this->process_meta(input, output);
            });
        oob_client_ = std::make_shared<EpollClient>();

        assert(oob_server_->start());
        assert(oob_client_->start());

    }

    // Destructor
    ~EFAEndpoint() {
        if (oob_client_) {
            oob_client_->stop();
        }
        if (oob_server_) {
            oob_server_->stop();
        }
    }

    // Getters
    int gpuIndex() const { return gpu_index_; }

    size_t contextCount() const { return contexts_.size(); }

    std::shared_ptr<RdmaContext> getContext(size_t index) const {
        if (index >= contexts_.size()) return nullptr;
        return contexts_[index];
    }

    const std::vector<std::shared_ptr<RdmaContext>>& contexts() const {
        return contexts_;
    }

    // Channel group management
    // std::shared_ptr<ChannelGroup> getChannelGroup(uint64_t connect_id) {
    //     auto it = channel_groups_.find(connect_id);
    //     if (it == channel_groups_.end()) return nullptr;
    //     return it->second;
    // }

    // void addChannelGroup(uint64_t connect_id, std::shared_ptr<ChannelGroup> group) {
    //     channel_groups_[connect_id] = group;
    // }

    // void createChannelGroup(uint64_t connect_id) {
    //     channel_groups_[connect_id] = std::make_shared<ChannelGroup>();
    // }

    // bool hasChannelGroup(uint64_t connect_id) const {
    //     return channel_groups_.find(connect_id) != channel_groups_.end();
    // }


    void addOneRecvChannel(uint64_t rank_id, std::shared_ptr<EFAChannel> new_channel){
        std::shared_ptr<ChannelGroup> group_ptr;
        auto it = recv_channel_groups_.find(rank_id);
        if (it == recv_channel_groups_.end()){
            group_ptr = std::make_shared<ChannelGroup>();
            recv_channel_groups_[rank_id] = group_ptr;
        }else{
            group_ptr = it->second;
        }
        group_ptr->addChannel(new_channel);
    }

    void addOneSendChannel(uint64_t rank_id, std::shared_ptr<EFAChannel> new_channel){
        std::shared_ptr<ChannelGroup> group_ptr;
        auto it = send_channel_groups_.find(rank_id);
        if (it == send_channel_groups_.end()){
            group_ptr = std::make_shared<ChannelGroup>();
            send_channel_groups_[rank_id] = group_ptr;
        }else{
            group_ptr = it->second;
        }
        group_ptr->addChannel(new_channel);
    }
    bool build_connect(uint64_t rank_id){
        if(rank_id == rank_id_){
            return false;
        }
        const auto& item = rank_oob_meta_.find(rank_id);
        if(item == rank_oob_meta_.end()){
            return false;
        }
        std::shared_ptr<OOBMetaData> ip_port_ptr = item->second;
        std::string oob_con;
        while(oob_con.empty()) {
            oob_con = oob_client_->connect_to_server(ip_port_ptr->server_ip, ip_port_ptr->server_port);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        std::cout<<"connected "<<std::endl;
        // std::vector<std::shared_ptr<EFAChannel>> 
        for(int i =0; i < contexts_.size(); i++){
            
            auto new_channel = std::make_shared<EFAChannel>(contexts_[i]);
            std::cout<<"?????????????"<< new_channel->get_local_meta()->qpn<<std::endl;
            addOneSendChannel(rank_id, new_channel);
            MetaInfoToExchange meta(rank_id_, i, new_channel->get_local_meta());
            std::string serialized_meta = serialize(meta);
            bool sent = oob_client_->send_meta(oob_con, serialized_meta, [new_channel](const std::string& response) {
                // Deserialize response as MetaInfoToExchange
                MetaInfoToExchange response_meta = deserialize<MetaInfoToExchange>(response);
                std::cout << "  rank_id: " << response_meta.rank_id << "\n";
                std::cout << "  context_id: " << response_meta.context_id << "\n";
                std::cout << "  channel_meta.qpn: " << response_meta.channel_meta.qpn << "\n";
                std::cout << "  mem_meta.addr: 0x" << std::hex << response_meta.mem_meta.addr << std::dec << "\n";
                std::cout << "  mem_meta.rkey: " << response_meta.mem_meta.rkey << "\n";
                std::cout << "  mem_meta.length: " << response_meta.mem_meta.length << "\n";
                new_channel->connect(response_meta.channel_meta);
            });
            
        }
        
        return true;
    }

    // Add or update rank OOB metadata from a given map
    void add_rank_oob_meta(const std::unordered_map<uint64_t, std::shared_ptr<OOBMetaData>>& new_meta) {
        for (const auto& [rank_id, meta_ptr] : new_meta) {
            rank_oob_meta_[rank_id] = meta_ptr;
        }
    }




private:
    // Initialize RDMA contexts based on device_ids
    void initializeContexts(const std::vector<size_t>& device_ids) {
        auto& device_manager = RdmaDeviceManager::instance();

        // Determine which devices to use
        std::vector<size_t> target_device_ids;
        if (device_ids.empty()) {
            // Use all available devices
            for (size_t i = 0; i < device_manager.deviceCount(); ++i) {
                target_device_ids.push_back(i);
            }
        } else {
            target_device_ids = device_ids;
        }

        // Create context for each device
        for (size_t device_id : target_device_ids) {
            auto device = device_manager.getDevice(device_id);
            if (!device) {
                std::cerr << "Warning: Device " << device_id << " not found, skipping" << std::endl;
                continue;
            }

            try {
                auto context = std::make_shared<RdmaContext>(device);
                contexts_.push_back(context);
                std::cout << "EFAEndpoint: Created context for device " << device_id
                         << " (" << device->name() << ")" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error creating context for device " << device_id
                         << ": " << e.what() << std::endl;
            }
        }

        if (contexts_.empty()) {
            throw std::runtime_error("EFAEndpoint: No contexts created");
        }
    }

    void process_meta(const std::string& input, std::string& output){
        std::shared_ptr<MetaInfoToExchange> meta = std::make_shared<MetaInfoToExchange>(deserialize<MetaInfoToExchange>(input));
        std::cout << "Processing: rank_id=" << meta->rank_id
                << " context_id=" << meta->context_id
                << " qpn=" << meta->channel_meta.qpn
                << " mem_addr=0x" << std::hex << meta->mem_meta.addr << std::dec
                << " rkey=" << meta->mem_meta.rkey << "\n";

        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        auto context_id = meta->context_id % contexts_.size();
        std::shared_ptr<RdmaContext> ctx_ptr = contexts_[context_id];
        std::shared_ptr<EFAChannel> new_channel = std::make_shared<EFAChannel>(ctx_ptr, meta->channel_meta);
        // Create response (echo back the same data)
        MetaInfoToExchange response(rank_id_, context_id ,new_channel->get_local_meta());
        std::cout<<"to send: "<< new_channel->get_local_meta()->qpn<<std::endl;
        output = serialize(response);
        addOneRecvChannel(meta->rank_id, new_channel);
    }


    uint64_t rank_id_;
    int gpu_index_;
    std::vector<std::shared_ptr<RdmaContext>> contexts_;
    std::unordered_map<uint64_t, std::shared_ptr<ChannelGroup>> recv_channel_groups_;
    std::unordered_map<uint64_t, std::shared_ptr<ChannelGroup>> send_channel_groups_;


    uint64_t port_;
    std::unordered_map<uint64_t, std::shared_ptr<OOBMetaData>> rank_oob_meta_;
    std::shared_ptr<EpollClient> oob_client_;
    std::shared_ptr<EpollServer> oob_server_;
};