// multiple_qp_example.cpp
// Example: Creating multiple QPs under the same ibv_context
// Compile: g++ -std=c++17 multiple_qp_example.cpp -o multiple_qp_example
// -libverbs -lefa

#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

struct qp_info {
  struct ibv_qp* qp;
  uint32_t qp_num;
  char const* name;
};

int main() {
  // 1. Get device list
  int num_devices;
  struct ibv_device** device_list = ibv_get_device_list(&num_devices);
  if (!device_list) {
    perror("Failed to get IB devices list");
    return 1;
  }

  printf("Found %d RDMA device(s)\n", num_devices);
  if (num_devices == 0) {
    fprintf(stderr, "No RDMA devices found\n");
    return 1;
  }

  // 2. Open the first device (single context)
  struct ibv_context* ctx = ibv_open_device(device_list[0]);
  if (!ctx) {
    perror("Failed to open device");
    ibv_free_device_list(device_list);
    return 1;
  }
  printf("Opened device: %s\n", ibv_get_device_name(device_list[0]));
  printf("Context pointer: %p\n", (void*)ctx);

  // 3. Query device capabilities
  struct ibv_device_attr device_attr;
  if (ibv_query_device(ctx, &device_attr)) {
    perror("Failed to query device");
    ibv_close_device(ctx);
    return 1;
  }
  printf("\nDevice Capabilities:\n");
  printf("  Max QPs: %d\n", device_attr.max_qp);
  printf("  Max CQs: %d\n", device_attr.max_cq);
  printf("  Max PDs: %d\n", device_attr.max_pd);

  // 4. Allocate Protection Domain (single PD shared by all QPs)
  struct ibv_pd* pd = ibv_alloc_pd(ctx);
  if (!pd) {
    perror("Failed to allocate PD");
    ibv_close_device(ctx);
    return 1;
  }
  printf("\nAllocated Protection Domain: %p\n", (void*)pd);

  // 5. Create Completion Queues (one for send, one for recv - shared by all
  // QPs)
  int cq_size = 128;
  struct ibv_cq* send_cq = ibv_create_cq(ctx, cq_size, NULL, NULL, 0);
  if (!send_cq) {
    perror("Failed to create send CQ");
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    return 1;
  }

  struct ibv_cq* recv_cq = ibv_create_cq(ctx, cq_size, NULL, NULL, 0);
  if (!recv_cq) {
    perror("Failed to create recv CQ");
    ibv_destroy_cq(send_cq);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    return 1;
  }
  printf("Created Completion Queues:\n");
  printf("  Send CQ: %p\n", (void*)send_cq);
  printf("  Recv CQ: %p\n", (void*)recv_cq);

  // 6. Create multiple QPs under the same context
  int num_qps = 5;  // Create 5 QPs
  std::vector<qp_info> qp_list;

  printf("\n--- Creating %d Queue Pairs under same context ---\n", num_qps);

  for (int i = 0; i < num_qps; i++) {
    // Initialize QP attributes
    struct ibv_qp_init_attr qp_init_attr = {};
    qp_init_attr.qp_type = IBV_QPT_RC;  // Reliable Connection
    qp_init_attr.sq_sig_all = 0;
    qp_init_attr.send_cq = send_cq;  // Shared send CQ
    qp_init_attr.recv_cq = recv_cq;  // Shared recv CQ
    qp_init_attr.cap.max_send_wr = 32;
    qp_init_attr.cap.max_recv_wr = 32;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;

    // Create QP
    struct ibv_qp* qp = ibv_create_qp(pd, &qp_init_attr);
    if (!qp) {
      perror("Failed to create QP");
      continue;
    }

    qp_info info;
    info.qp = qp;
    info.qp_num = qp->qp_num;
    info.name = (i == 0)   ? "QP_0 (Control)"
                : (i == 1) ? "QP_1 (Data-High)"
                : (i == 2) ? "QP_2 (Data-Medium)"
                : (i == 3) ? "QP_3 (Data-Low)"
                           : "QP_4 (Backup)";
    qp_list.push_back(info);

    printf("  QP[%d]: Created %s\n", i, info.name);
    printf("    - QP Number: %u\n", info.qp_num);
    printf("    - QP Pointer: %p\n", (void*)qp);
    printf("    - QP Context: %p (same as device context)\n",
           (void*)qp->context);
    printf("    - PD: %p (shared)\n", (void*)qp->pd);
    printf("    - Send CQ: %p (shared)\n", (void*)qp->send_cq);
    printf("    - Recv CQ: %p (shared)\n", (void*)qp->recv_cq);
  }

  printf("\n--- Summary ---\n");
  printf("Single ibv_context (%p) supports:\n", (void*)ctx);
  printf("  - %zu Queue Pairs created successfully\n", qp_list.size());
  printf("  - All QPs share the same PD (%p)\n", (void*)pd);
  printf("  - All QPs share the same Send CQ (%p)\n", (void*)send_cq);
  printf("  - All QPs share the same Recv CQ (%p)\n", (void*)recv_cq);

  // 7. Example: You can also create QP_EX (Extended QP)
  printf("\n--- Creating Extended QP (QP_EX) ---\n");

  // First create CQ_EX
  struct ibv_cq_init_attr_ex cq_attr_ex = {};
  cq_attr_ex.cqe = 128;
  cq_attr_ex.wc_flags = IBV_WC_EX_WITH_BYTE_LEN | IBV_WC_EX_WITH_IMM;

  struct ibv_cq_ex* cq_ex = ibv_create_cq_ex(ctx, &cq_attr_ex);
  if (cq_ex) {
    printf("Created CQ_EX: %p\n", (void*)cq_ex);

    // Create QP_EX
    struct ibv_qp_init_attr_ex qp_attr_ex = {};
    qp_attr_ex.qp_type = IBV_QPT_DRIVER;  // For EFA
    qp_attr_ex.comp_mask =
        IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
    qp_attr_ex.pd = pd;
    qp_attr_ex.send_cq = ibv_cq_ex_to_cq(cq_ex);
    qp_attr_ex.recv_cq = ibv_cq_ex_to_cq(cq_ex);
    qp_attr_ex.cap.max_send_wr = 32;
    qp_attr_ex.cap.max_recv_wr = 32;
    qp_attr_ex.cap.max_send_sge = 1;
    qp_attr_ex.cap.max_recv_sge = 1;
    qp_attr_ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE | IBV_QP_EX_WITH_SEND;

    struct ibv_qp* qp_ex = ibv_create_qp_ex(ctx, &qp_attr_ex);
    if (qp_ex) {
      printf("Created QP_EX successfully!\n");
      printf("  - QP_EX Number: %u\n", qp_ex->qp_num);
      printf("  - QP_EX Pointer: %p\n", (void*)qp_ex);
      printf("  - Same Context: %p\n", (void*)qp_ex->context);

      // Add to list
      qp_info info;
      info.qp = qp_ex;
      info.qp_num = qp_ex->qp_num;
      info.name = "QP_EX (Extended)";
      qp_list.push_back(info);

      // Clean up qp_ex
      ibv_destroy_qp(qp_ex);
    } else {
      printf("Note: QP_EX creation might fail on some devices\n");
    }

    // Clean up cq_ex
    ibv_destroy_cq(ibv_cq_ex_to_cq(cq_ex));
  }

  // 8. Use case examples
  printf("\n--- Typical Use Cases for Multiple QPs ---\n");
  printf("1. Multi-connection to different peers:\n");
  printf("   QP[0] <-> Remote Node A\n");
  printf("   QP[1] <-> Remote Node B\n");
  printf("   QP[2] <-> Remote Node C\n");
  printf("\n");
  printf("2. QoS / Priority lanes to same peer:\n");
  printf("   QP[0] = High priority traffic\n");
  printf("   QP[1] = Normal priority traffic\n");
  printf("   QP[2] = Low priority / background traffic\n");
  printf("\n");
  printf("3. Multi-rail for bandwidth aggregation:\n");
  printf("   QP[0] + QP[1] + QP[2] -> Same destination (parallel transfer)\n");
  printf("\n");
  printf("4. Different transport types:\n");
  printf("   QP[0] = RC (Reliable Connection)\n");
  printf("   QP[1] = UC (Unreliable Connection)\n");
  printf("   QP[2] = UD (Unreliable Datagram)\n");

  // 9. Cleanup - destroy all QPs
  printf("\n--- Cleaning up ---\n");
  for (size_t i = 0; i < qp_list.size(); i++) {
    if (ibv_destroy_qp(qp_list[i].qp) == 0) {
      printf("Destroyed %s\n", qp_list[i].name);
    }
  }

  ibv_destroy_cq(recv_cq);
  ibv_destroy_cq(send_cq);
  ibv_dealloc_pd(pd);
  ibv_close_device(ctx);
  ibv_free_device_list(device_list);

  printf(
      "\nConclusion: YES, you can create multiple QPs and QP_EX under same "
      "context!\n");
  return 0;
}
