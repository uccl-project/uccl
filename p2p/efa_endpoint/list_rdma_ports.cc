#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    int num_devices;
    struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) {
        perror("Failed to get IB devices list");
        return 1;
    }

    printf("Found %d RDMA devices:\n", num_devices);
    for (int i = 0; i < num_devices; ++i) {
        printf("[%d] Device name: %s\n", i, ibv_get_device_name(dev_list[i]));

        struct ibv_context *ctx = ibv_open_device(dev_list[i]);
        if (!ctx) continue;

        struct ibv_device_attr attr;
        if (ibv_query_device(ctx, &attr)) {
            perror("ibv_query_device failed");
            continue;
        }

        printf("  -> This device has %d port(s)\n", attr.phys_port_cnt);

        for (uint8_t port_num = 1; port_num <= attr.phys_port_cnt; ++port_num) {
            struct ibv_port_attr port_attr;
            if (!ibv_query_port(ctx, port_num, &port_attr)) {
                printf("     Port %d state: %s, LID: %d\n",
                       port_num,
                       (port_attr.state == IBV_PORT_ACTIVE) ? "ACTIVE" : "DOWN",
                       port_attr.lid);
            }
        }

        ibv_close_device(ctx);
    }

    ibv_free_device_list(dev_list);
    return 0;
}
