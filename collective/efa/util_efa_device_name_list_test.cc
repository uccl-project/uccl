#ifndef UCCL_TESTING
#define UCCL_TESTING
#endif

#include "util_efa.h"
#include <gtest/gtest.h>
#include <cstdlib>

using uccl::GetEfaDeviceNameList;
using uccl::GetEnaDeviceNameList;
using uccl::_detail::ResetDeviceNameListsForTest;

class EfaDeviceNameListTest : public ::testing::Test {
 protected:
  void SetUp() override { ResetDeviceNameListsForTest(); }
  void TearDown() override { unsetenv("UCCL_EFA_DEVICES"); }
};

class EnaDevice : public EfaDeviceNameListTest {
 protected:
  void SetUp() override { ResetDeviceNameListsForTest(); }
  void TearDown() override { unsetenv("UCCL_ENA_DEVICES"); }
};

// 1. Set env var EFA devices
TEST_F(EfaDeviceNameListTest, RespectsEnvVariable) {
  setenv("UCCL_EFA_DEVICES", "efa0,efa1, rdmap42", /*overwrite=*/1);

  auto const& list = GetEfaDeviceNameList();
  EXPECT_EQ(list, std::vector<std::string>({"efa0", "efa1", "rdmap42"}));
}

// 2. Set env var ENA devices
TEST_F(EnaDevice, RespectsEnvVariable) {
  setenv("UCCL_ENA_DEVICES", "ena0,ena1, ens42", /*overwrite=*/1);

  auto const& list = GetEnaDeviceNameList();
  EXPECT_EQ(list, std::vector<std::string>({"ena0", "ena1", "ens42"}));
}
