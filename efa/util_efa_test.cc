#include "util_efa.h"
#include "transport_config.h"
#include "util.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>
#include <pthread.h>
#include <signal.h>

using namespace uccl;

#define TCP_PORT 12345  // Port for exchanging QPNs & GIDs
#define ITERATIONS 10240
#define MAX_INFLIGHT 1024u
#define EFA_DEV_ID 0
#define GPU_ID (EFA_DEV_ID * 2)
#define SOCKET_ID (EFA_DEV_ID * 2)
// #define RESPONDE_ACK

// Exchange QPNs and GIDs via TCP
void exchange_qpns(char const* peer_ip, ConnMeta* local_metadata,
                   ConnMeta* remote_metadata) {
  int sock;
  struct sockaddr_in addr;
  char mode = peer_ip ? 'c' : 's';

  sock = socket(AF_INET, SOCK_STREAM, 0);
  int opt = 1;
  setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt,
             sizeof(opt));  // Avoid port conflicts

  addr.sin_family = AF_INET;
  addr.sin_port = htons(TCP_PORT);
  addr.sin_addr.s_addr = peer_ip ? inet_addr(peer_ip) : INADDR_ANY;

  if (mode == 's') {
    printf("Server waiting for connection...\n");
    bind(sock, (struct sockaddr*)&addr, sizeof(addr));
    listen(sock, 128);
    sock = accept(sock, NULL, NULL);  // Blocks if no client
    printf("Server accepted connection\n");
  } else {
    printf("Client attempting connection...\n");
    while (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
      perror("Connect failed, retrying...");
      sleep(1);
    }
    printf("Client connected\n");
  }

  // Set receive timeout to avoid blocking
  struct timeval timeout = {5, 0};  // 5 seconds timeout
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

  // Send local QPN and GID
  if (send(sock, local_metadata, sizeof(ConnMeta), 0) <= 0)
    perror("send() failed");

  // Receive remote QPN and GID
  if (recv(sock, remote_metadata, sizeof(ConnMeta), 0) <= 0)
    perror("recv() timeout");

  close(sock);
  printf("QPNs and GIDs exchanged\n");
}

std::vector<FrameDesc*> prepare_frames(EFASocket* socket,
                                       struct ibv_ah* dest_ah,
                                       ConnMeta* remote_meta, int num_frames) {
  std::vector<FrameDesc*> frames;
  for (int i = 0; i < num_frames; i++) {
    auto frame_desc = socket->pop_frame_desc();
    auto pkt_hdr = socket->pop_pkt_hdr();
    auto pkt_data = socket->pop_pkt_data();
    auto frame =
        FrameDesc::Create(frame_desc, pkt_hdr, kUcclPktHdrLen, pkt_data,
                          kUcclPktDataMaxLen, socket->get_pkt_data_lkey(), 0);
    frame->set_dest_ah(dest_ah);
    auto dst_qp_idx = IntRand(0, kMaxDstQP - 1);
    frame->set_dest_qpn(remote_meta->qpn_list[dst_qp_idx]);
    frames.push_back(frame);
  }
  return frames;
}

std::vector<FrameDesc*> prepare_frames_for_ctrl(EFASocket* socket,
                                                struct ibv_ah* dest_ah,
                                                ConnMeta* remote_meta,
                                                int num_frames) {
  std::vector<FrameDesc*> frames;
  for (int i = 0; i < num_frames; i++) {
    auto frame_desc = socket->pop_frame_desc();
    auto pkt_hdr = socket->pop_pkt_hdr();
    auto frame = FrameDesc::Create(
        frame_desc, pkt_hdr, kUcclPktHdrLen + kUcclSackHdrLen, 0, 0, 0, 0);
    frame->set_dest_ah(dest_ah);
    auto dst_qp_idx = IntRand(0, kMaxDstQPCtrl - 1);
    frame->set_dest_qpn(remote_meta->qpn_list_ctrl[dst_qp_idx]);
    frames.push_back(frame);
  }
  return frames;
}

void run_server() {
  auto* socket = EFAFactory::CreateSocket(GPU_ID, EFA_DEV_ID, SOCKET_ID);

  auto local_meta = new ConnMeta();
  socket->get_conn_metadata(local_meta);
  auto remote_meta = new ConnMeta();
  exchange_qpns(nullptr, local_meta, remote_meta);

  auto* dev = EFAFactory::GetEFADevice(EFA_DEV_ID);
  auto* dest_ah = dev->create_ah(remote_meta->gid);

  std::vector<FrameDesc*> frames;
  FrameDesc* frame;

  TimePoint start_time, mid_time, end_time;
  std::chrono::duration<double> duration1, duration2;
  for (int i = 0; i < ITERATIONS; i++) {
    // Receiving a data packet.
    do {
      frames = socket->poll_recv_cq(1);
    } while (frames.size() == 0);

    start_time = std::chrono::high_resolution_clock::now();

    CHECK(frames.size() == 1);
    frame = frames[0];
    CHECK(strcmp((char*)(frame->get_pkt_hdr_addr() + EFA_UD_ADDITION),
                 "Hello World") == 0);
    socket->push_pkt_hdr(frame->get_pkt_hdr_addr());
    socket->push_pkt_data(frame->get_pkt_data_addr());
    socket->push_frame_desc((uint64_t)frame);
    CHECK(socket->recv_queue_wrs() >=
          kMaxRecvWr * kMaxDstQP - kMaxRecvWrDeficit * kMaxDstQP);

    // Send it back.
    frame =
        FrameDesc::Create(socket->pop_frame_desc(), socket->pop_pkt_hdr(),
                          kUcclPktHdrLen, socket->pop_pkt_data(),
                          kUcclPktDataMaxLen, socket->get_pkt_data_lkey(), 0);
    strcpy((char*)frame->get_pkt_hdr_addr(), "Hello World");
    frame->set_dest_ah(dest_ah);
    frame->set_dest_qpn(remote_meta->qpn_list[0]);
    socket->post_send_wr(frame, socket->get_next_src_qp_idx_for_send());

    mid_time = std::chrono::high_resolution_clock::now();
    duration1 += mid_time - start_time;

    do {
      frames = socket->poll_send_cq(1);
    } while (frames.size() == 0);
    CHECK(frames.size() == 1);
    frame = frames[0];
    socket->push_pkt_hdr(frame->get_pkt_hdr_addr());
    socket->push_pkt_data(frame->get_pkt_data_addr());
    socket->push_frame_desc((uint64_t)frame);
    CHECK_EQ(socket->send_queue_free_space(), kMaxSendWr * kMaxSrcQP);

    end_time = std::chrono::high_resolution_clock::now();
    duration2 += end_time - mid_time;
  }

  LOG(INFO) << "post_send_wr duration "
            << std::chrono::duration_cast<std::chrono::microseconds>(duration1)
                       .count() *
                   1.0 / ITERATIONS
            << " us"
            << " poll_send_cq duration "
            << std::chrono::duration_cast<std::chrono::microseconds>(duration2)
                       .count() *
                   1.0 / ITERATIONS
            << " us";

  // Receiving an ack ctrl packet.
  uint32_t polled_send_acks = 0;
  do {
    std::tie(frames, polled_send_acks) = socket->poll_ctrl_cq(1);
  } while (frames.size() == 0);
  CHECK(frames.size() == 1 && polled_send_acks == 0);
  frame = frames[0];
  CHECK(strcmp((char*)(frame->get_pkt_hdr_addr() + EFA_UD_ADDITION),
               "Ctrl Packet") == 0);
  socket->push_pkt_hdr(frame->get_pkt_hdr_addr());
  socket->push_frame_desc((uint64_t)frame);
  CHECK(socket->recv_queue_wrs() >=
        kMaxRecvWr * kMaxDstQP - kMaxRecvWrDeficit * kMaxDstQP);

  // Benchmarking throughput
  int i = 0;
  while (i < ITERATIONS) {
    // Receiving data packets
    frames = socket->poll_recv_cq(RECV_BATCH_SIZE);
    VLOG(4) << "Received " << frames.size() << " frames";
    for (auto frame : frames) {
      socket->push_pkt_hdr(frame->get_pkt_hdr_addr());
      socket->push_pkt_data(frame->get_pkt_data_addr());
      socket->push_frame_desc((uint64_t)frame);
    }
    auto recv_frames = frames.size();

#ifdef RESPONDE_ACK
    // Sending ack ctrl packets
    frames = prepare_frames_for_ctrl(socket, dest_ah, remote_meta, recv_frames);
    socket->post_send_wrs_for_ctrl(frames,
                                   socket->get_next_src_qp_idx_for_send_ctrl());
    VLOG(4) << "Sent " << frames.size() << " acks";

    // Check if any ack ctrl packet finishes sending.
    std::tie(frames, polled_send_acks) = socket->poll_ctrl_cq(RECV_BATCH_SIZE);
    CHECK_EQ(frames.size(), 0);
    VLOG(4) << "Polled " << polled_send_acks << " sent acks";
#else
    // Sending data packets back
    frames = prepare_frames(socket, dest_ah, remote_meta, recv_frames);
    socket->post_send_wrs(frames, socket->get_next_src_qp_idx_for_send());
    VLOG(4) << "Sent " << frames.size() << " frames";

    // Check if any send finished.
    frames = socket->poll_send_cq(RECV_BATCH_SIZE);
    for (auto frame : frames) {
      socket->push_pkt_hdr(frame->get_pkt_hdr_addr());
      socket->push_pkt_data(frame->get_pkt_data_addr());
      socket->push_frame_desc((uint64_t)frame);
    }
    VLOG(4) << "Polled " << frames.size() << " send frames";
#endif

    i += recv_frames;
  }
}

void run_client(char const* server_ip) {
  auto* socket = EFAFactory::CreateSocket(GPU_ID, EFA_DEV_ID, SOCKET_ID);

  auto local_meta = new ConnMeta();
  socket->get_conn_metadata(local_meta);
  auto remote_meta = new ConnMeta();
  exchange_qpns(server_ip, local_meta, remote_meta);

  auto* dev = EFAFactory::GetEFADevice(EFA_DEV_ID);
  auto* dest_ah = dev->create_ah(remote_meta->gid);

  uint64_t frame_desc, pkt_hdr, pkt_data;
  FrameDesc* frame;
  std::vector<FrameDesc*> frames;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < ITERATIONS; i++) {
    // Send a data packet
    frame_desc = socket->pop_frame_desc();
    pkt_hdr = socket->pop_pkt_hdr();
    pkt_data = socket->pop_pkt_data();
    frame =
        FrameDesc::Create(frame_desc, pkt_hdr, kUcclPktHdrLen, pkt_data,
                          kUcclPktDataMaxLen, socket->get_pkt_data_lkey(), 0);
    frame->set_dest_ah(dest_ah);
    frame->set_dest_qpn(remote_meta->qpn_list[0]);
    strcpy((char*)pkt_hdr, "Hello World");
    socket->post_send_wr(frame, socket->get_next_src_qp_idx_for_send());
    do {
      frames = socket->poll_send_cq(1);
    } while (frames.size() == 0);
    CHECK(frames.size() == 1);
    CHECK(frames[0] == frame);
    socket->push_frame_desc((uint64_t)frame);
    socket->push_pkt_hdr(pkt_hdr);
    socket->push_pkt_data(pkt_data);
    CHECK_EQ(socket->send_queue_free_space(), kMaxSendWr * kMaxSrcQP);

    // Receiving the packet back
    do {
      frames = socket->poll_recv_cq(1);
    } while (frames.size() == 0);
    CHECK(frames.size() == 1);
    frame = frames[0];
    CHECK(strcmp((char*)(frame->get_pkt_hdr_addr() + EFA_UD_ADDITION),
                 "Hello World") == 0);
    socket->push_frame_desc((uint64_t)frame);
    socket->push_pkt_hdr(pkt_hdr);
    socket->push_pkt_data(pkt_data);
    CHECK(socket->recv_queue_wrs() >=
          kMaxRecvWr * kMaxDstQP - kMaxRecvWrDeficit * kMaxDstQP);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  LOG(INFO) << "Round trip time: " << duration_us.count() * 1.0 / ITERATIONS
            << " us";

  // Sending a ctrl packet
  frame_desc = socket->pop_frame_desc();
  pkt_hdr = socket->pop_pkt_hdr();
  frame = FrameDesc::Create(frame_desc, pkt_hdr,
                            kUcclPktHdrLen + kUcclSackHdrLen, 0, 0, 0, 0);
  frame->set_dest_ah(dest_ah);
  frame->set_dest_qpn(remote_meta->qpn_list_ctrl[0]);  // ctrl qp
  strcpy((char*)pkt_hdr, "Ctrl Packet");
  socket->post_send_wr(frame, socket->get_next_src_qp_idx_for_send_ctrl());
  uint32_t polled_send_acks = 0;
  do {
    std::tie(frames, polled_send_acks) = socket->poll_ctrl_cq(1);
  } while (polled_send_acks == 0);
  CHECK(polled_send_acks == 1 && frames.size() == 0);
  // No need to push pkt_hdr and frame_desc, as poll_ctrl_cq() frees them.
  CHECK_EQ(socket->send_queue_free_space(), kMaxSendWr * kMaxSrcQP);

  // Benchmarking throughput
  int i = 0;
  int inflights = 0;

  start_time = std::chrono::high_resolution_clock::now();
  while (i < ITERATIONS) {
    VLOG(4) << "ITERATIONS i: " << i << ", inflights: " << inflights;
    // Send data packets if allowed.
    if (inflights < MAX_INFLIGHT) {
      auto allowed_frames = std::min(MAX_INFLIGHT - inflights, SEND_BATCH_SIZE);
      frames = prepare_frames(socket, dest_ah, remote_meta, allowed_frames);
      socket->post_send_wrs(frames, socket->get_next_src_qp_idx_for_send());

      i += frames.size();
      inflights += frames.size();
      VLOG(4) << "Sent " << frames.size()
              << " frames, inflights: " << inflights;
    }

    // Check if any send finished.
    frames = socket->poll_send_cq(SEND_BATCH_SIZE);
    for (auto frame : frames) {
      socket->push_pkt_hdr(frame->get_pkt_hdr_addr());
      socket->push_pkt_data(frame->get_pkt_data_addr());
      socket->push_frame_desc((uint64_t)frame);
    }
    VLOG(4) << "Polled " << frames.size() << " send frames";

#ifdef RESPONDE_ACK
    // Check if any ack received.
    std::tie(frames, polled_send_acks) = socket->poll_ctrl_cq(RECV_BATCH_SIZE);
    for (auto frame : frames) {
      socket->push_pkt_hdr(frame->get_pkt_hdr_addr());
      DCHECK_EQ(frame->get_pkt_data_len(), 0) << frame->get_pkt_hdr_len();
      socket->push_frame_desc((uint64_t)frame);
    }
    CHECK_EQ(polled_send_acks, 0);
#else
    // Check if any data packets received.
    frames = socket->poll_recv_cq(RECV_BATCH_SIZE);
    for (auto frame : frames) {
      socket->push_pkt_hdr(frame->get_pkt_hdr_addr());
      socket->push_pkt_data(frame->get_pkt_data_addr());
      socket->push_frame_desc((uint64_t)frame);
    }
#endif

    inflights -= frames.size();
    VLOG(4) << "Polled " << frames.size()
            << " frames, inflights: " << inflights;
  }
  end_time = std::chrono::high_resolution_clock::now();

  duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  LOG(INFO) << "Throughput: " << ITERATIONS * 1.0 / duration_us.count()
            << " Mops/s "
            << " bandwidth: "
            << ITERATIONS * 1.0 / duration_us.count() * EFA_MTU * 8 / 1000
            << " Gbps";
}

// TO RUN THE TEST:
// On server: ./util_efa_test
// On client: ./util_efa_test server_ip
int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::once_flag init_flag;
  std::call_once(init_flag, []() { EFAFactory::Init(0); });

  if (argc == 1) {
    // Server
    run_server();
  } else {
    // Client
    run_client(argv[1]);
  }

  return 0;
}
