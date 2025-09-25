#include "../tcpx_interface.h"
#include <cstdlib>
#include <iostream>

int main() {
  std::cout << "=== TCPX Device Discovery Test ===" << std::endl;

  // Enable debug output
  setenv("UCCL_TCPX_DEBUG", "1", 1);

  std::cout << "\n[Step 1] Testing TCPX device discovery..." << std::endl;

  // Test device discovery
  int device_count = tcpx_get_device_count();
  std::cout << "Device count result: " << device_count << std::endl;

  if (device_count > 0) {
    std::cout << "✓ SUCCESS: Found " << device_count << " TCPX devices"
              << std::endl;

    std::cout << "\n[Step 2] Testing multiple calls to device discovery..."
              << std::endl;
    // Test that subsequent calls work (plugin should already be loaded)
    int device_count2 = tcpx_get_device_count();
    if (device_count2 == device_count) {
      std::cout << "✓ SUCCESS: Consistent device count on second call"
                << std::endl;
    } else {
      std::cout << "⚠ WARNING: Device count changed: " << device_count2
                << std::endl;
    }

    std::cout << "\n=== TCPX Device Discovery Test PASSED ===" << std::endl;
    std::cout << "Ready to proceed with connection testing!" << std::endl;
    return 0;

  } else if (device_count == 0) {
    std::cout << "⚠ WARNING: Plugin loaded but no TCPX devices found"
              << std::endl;
    std::cout << "This might be expected if no TCPX hardware is available"
              << std::endl;
    return 0;

  } else {
    std::cout << "✗ FAILED: Device discovery returned error: " << device_count
              << std::endl;
    std::cout << "\nTroubleshooting steps:" << std::endl;
    std::cout << "1. Check if TCPX plugin exists:" << std::endl;

    char const* plugin_path = getenv("UCCL_TCPX_PLUGIN_PATH");
    if (!plugin_path) {
      plugin_path = "/usr/local/tcpx/lib64/libnccl-net-tcpx.so";
    }
    std::cout << "   Plugin path: " << plugin_path << std::endl;

    std::cout << "2. Try setting UCCL_TCPX_PLUGIN_PATH to correct path"
              << std::endl;
    std::cout << "3. Ensure TCPX plugin is properly installed" << std::endl;
    std::cout << "4. Check plugin dependencies (CUDA, network drivers)"
              << std::endl;

    return 1;
  }
}
