#include "queue_fixture.hpp"

using namespace local_test;

__host__ __device__ TransferCmd command(uint32_t id) {
  TransferCmd cmd{};
  cmd.cmd_type = make_cmd_type(id % 2 ? CmdType::ATOMIC : CmdType::WRITE,
                               (id & 2) != 0, (id & 4) != 0);
  cmd.dst_rank = id % 256;
  cmd.bytes_and_val = 0x12345678u ^ id;
  cmd.req_rptr = id;
  if (id % 2)
    cmd.value = -static_cast<int>(id + 1);
  else
    cmd.req_lptr = 0x87654321u ^ id;
  cmd.expert_idx = static_cast<uint16_t>(0xbadcu ^ id);
  return cmd;
}

__global__ void publish(uccl_gin::UCCLGinResources resources, int commands) {
  int const producer = blockIdx.x * blockDim.x + threadIdx.x;
  int const producers = gridDim.x * blockDim.x;
  for (int id = producer; id < commands; id += producers) {
    resources.d2h_queues[0]->atomic_set_and_commit(command(id));
  }
}

__global__ void scalar_flush(uccl_gin::UCCLGinResources resources) {
  uccl_gin::UCCLGin(resources).flush();
}

int main(int argc, char** argv) {
  int const device = integer(argc, argv, "--device", 0);
  int const producers = integer(argc, argv, "--producers", 32);
  int const capacity = integer(argc, argv, "--capacity", 512);
  int const commands = integer(argc, argv, "--commands", 100000);
  int const rounds = integer(argc, argv, "--rounds", 3);
  require(producers > 0 && producers <= capacity && producers <= 1024,
          "producer participation must fit the FIFO and one CUDA block");
  require(commands > 0 && rounds > 0, "empty test workload");
  QueueFixture fixture(device, 1, capacity);
  for (int round = 0; round < rounds; ++round) {
    std::vector<uint8_t> seen(commands);
    uint64_t received = 0, quiet = 0;
    fixture.start([&](size_t queue, TransferCmd const& actual) {
      if (get_base_cmd(actual.cmd_type) == CmdType::QUIET) {
        ++quiet;
        return;
      }
      uint32_t const id = actual.req_rptr;
      require(queue == 0 && id < seen.size(), "invalid decoded command id");
      require(seen[id] == 0, "duplicate command");
      auto const expected = command(id);
      require(std::memcmp(&actual, &expected, sizeof(actual)) == 0,
              "decoded command differs from producer payload");
      seen[id] = 1;
      ++received;
    });
    auto const start = std::chrono::steady_clock::now();
    publish<<<1, producers, 0, fixture.stream>>>(fixture.resources, commands);
    scalar_flush<<<1, 1, 0, fixture.stream>>>(fixture.resources);
    fixture.finish();
    double const ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    require(received == static_cast<uint64_t>(commands) && quiet == 1,
            "missing commands or incorrect scalar flush count");
    std::printf(
        "{\"test\":\"queue_smoke\",\"device\":%d,\"round\":%d,"
        "\"producers\":%d,\"capacity\":%d,\"commands\":%llu,"
        "\"quiet\":%llu,\"elapsed_ms\":%.3f,\"pass\":true}\n",
        device, round, producers, capacity,
        static_cast<unsigned long long>(received),
        static_cast<unsigned long long>(quiet), ms);
  }
}
