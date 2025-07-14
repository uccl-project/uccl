#define UCCL_TESTING     // must precede any header that pulls in util_efa.h
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


// 1. Env-var path
TEST_F(EfaDeviceNameListTest, RespectsEnvVariable) {
  setenv("UCCL_EFA_DEVICES", "efa0,efa1, rdmap42", /*overwrite=*/1);

  const auto& list = GetEfaDeviceNameList();
  EXPECT_EQ(list, std::vector<std::string>({"efa0", "efa1", "rdmap42"}));
}

// 1.1 Env-var path ENA
TEST_F(EnaDevice, RespectsEnvVariable) {
  setenv("UCCL_ENA_DEVICES", "ena0,ena1, ens42", /*overwrite=*/1);

  const auto& list = GetEnaDeviceNameList();
  EXPECT_EQ(list, std::vector<std::string>({"ena0", "ena1", "ens42"}));
}

// 2. Fallback path (no env, no real hw)
//    We don’t enumerate hardware in a unit test, so we just make sure
//    the function returns the built-in defaults when env is empty.
TEST_F(EfaDeviceNameListTest, UsesBuiltInDefaultsWhenEnvMissing) {
  // Explicitly ensure environment variables are not set to force hardware enumeration
  unsetenv("UCCL_EFA_DEVICES");
  unsetenv("UCCL_ENA_DEVICES");

  // Check EFA defaults
  const auto& efa_list = GetEfaDeviceNameList();
  std::vector<std::string> kExpectedEfaDefaults = {"rdmap16s27", "rdmap32s27", "rdmap144s27", "rdmap160s27"};
  for (const auto& name : kExpectedEfaDefaults)
    EXPECT_NE(std::find(efa_list.begin(), efa_list.end(), name), efa_list.end()) << "Missing EFA default: " << name;

  // Check ENA defaults
  const auto& ena_list = GetEnaDeviceNameList();
  std::vector<std::string> kExpectedEnaDefaults = {"ens32", "ens65", "ens130", "ens163"};
  for (const auto& name : kExpectedEnaDefaults)
    EXPECT_NE(std::find(ena_list.begin(), ena_list.end(), name), ena_list.end()) << "Missing ENA default: " << name;
}
