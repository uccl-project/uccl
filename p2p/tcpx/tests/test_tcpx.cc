#include "../tcpx_interface.h"
#include <stdio.h>

int main() {
  printf("🧪 TCPX quick test\n");
  printf("==================================================\n");

  // Test plugin loading
  printf("🔄 Loading the TCPX plugin...\n");
  int result = tcpx_load_plugin("/usr/local/tcpx/lib64/libnccl-net-tcpx.so");
  if (result == 0) {
    printf("✅ TCPX plugin loaded successfully\n");
  } else {
    printf("❌ Failed to load the TCPX plugin\n");
  }

  // Test device discovery
  printf("🔄 Querying TCPX device count...\n");
  int device_count = tcpx_get_device_count();
  printf("📊 Detected %d TCPX devices\n", device_count);

  printf("==================================================\n");
  if (result == 0 && device_count > 0) {
    printf("🎉 TCPX basic functionality test passed!\n");
    return 0;
  } else {
    printf("❌ TCPX basic functionality test failed\n");
    return 1;
  }
}
