#include <iostream>
#include "seq_num.h"
int main() {
    BitmapPacketTracker tracker;
    
    // 发送一些包
    std::vector<uint32_t> sent_packets;
    for (int i = 0; i < 10; ++i) {
        uint32_t seq = tracker.sendPacket();
        sent_packets.push_back(seq);
        std::cout << "Sent packet with seq: " << seq << std::endl;
    }
    
    // 模拟确认一些包
    tracker.acknowledge(sent_packets[0]);
    tracker.acknowledge(sent_packets[2]);
    tracker.acknowledge(sent_packets[4]);
    
    // 检查状态
    std::cout << "Unacknowledged packets: ";
    for (auto seq : tracker.getUnacknowledgedPackets()) {
        std::cout << seq << " ";
    }
    std::cout << std::endl;
    
    // 检查特定包
    std::cout << "Packet 3 acknowledged: " << tracker.isAcknowledged(3) << std::endl;
    std::cout << "Packet 4 acknowledged: " << tracker.isAcknowledged(4) << std::endl;
    
    return 0;
}