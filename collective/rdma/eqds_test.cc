#include "eqds.h"
#include <gtest/gtest.h>
#include <array>
#include <vector>
#include <unistd.h>

namespace uccl {
namespace eqds {
namespace {

class EQDSPacerTest : public ::testing::TestWithParam<size_t> {
 protected:
  void SetUp() override {
    // Stop the empty pacer before driving its public operations synchronously.
    pacer_.shutdown();
    context_.ops.post_send = [](ibv_qp*, ibv_send_wr*, ibv_send_wr**) {
      return 0;
    };
    qp_.context = &context_;
    qpw_.credit_qp_ = &qp_;
    qpw_.pacer_credit_chunk_pool_ = &pool_;
    qpw_.poll_item.pc_qpw = &qpw_;
    INIT_LIST_HEAD(&qpw_.poll_item.poll_link);

    uint64_t buffer;
    while (pool_.alloc_buff(&buffer) == 0) held_buffers_.push_back(buffer);
  }

  void RegisterIdleSenders(size_t count) {
    for (size_t i = 0; i < count; ++i) {
      auto& sender = senders_[i];
      sender.init_active_item();
      sender.init_idle_item();
      sender.latest_pull_ = 50;
      sender.highest_pull_target_.store(50);
      sender.set_fid(i);
      sender.set_pacer_credit_qpw(&qpw_);
      pacer_.request_pull(&sender);
    }
    pacer_.handle_pull_request();
    // Zero-backlog senders become idle without consuming credit buffers.
    pacer_.handle_grant_credit();
  }

  void RestoreBuffers() {
    for (auto buffer : held_buffers_) pool_.free_buff(buffer);
    held_buffers_.clear();
  }

  CreditChunkBuffPool pool_{nullptr};
  ibv_context context_{};
  ibv_qp qp_{};
  PacerCreditQPWrapper qpw_{};
  std::array<EQDSCC, 3> senders_{};
  std::vector<uint64_t> held_buffers_;
  EQDS pacer_{0, 50e9};
};

TEST_P(EQDSPacerTest, GrantsEachIdleSenderOncePerPass) {
  auto count = GetParam();
  RegisterIdleSenders(count);
  RestoreBuffers();
  auto available = pool_.size();

  pacer_.handle_grant_credit();
  for (size_t i = 0; i < count; ++i) {
    // One pull grants four quanta; 54 - 50 is below the idle limit of 30.
    EXPECT_EQ(senders_[i].latest_pull_, 54);
    EXPECT_FALSE(list_empty(&senders_[i].idle_item.idle_link));
  }
  EXPECT_EQ(pool_.size(), available - count);

  pacer_.handle_grant_credit();
  for (size_t i = 0; i < count; ++i) {
    EXPECT_EQ(senders_[i].latest_pull_, 58);
    EXPECT_FALSE(list_empty(&senders_[i].idle_item.idle_link));
  }
  EXPECT_EQ(pool_.size(), available - 2 * count);
}

INSTANTIATE_TEST_SUITE_P(SenderCounts, EQDSPacerTest,
                         ::testing::Values(0, 1, 2, 3));

TEST_F(EQDSPacerTest, StopsAfterTheOriginalTailIsRemoved) {
  RegisterIdleSenders(3);
  // The original tail reaches 32 surplus quanta and leaves the idle list.
  senders_[2].latest_pull_ = 78;
  RestoreBuffers();

  pacer_.handle_grant_credit();

  EXPECT_EQ(senders_[0].latest_pull_, 54);
  EXPECT_EQ(senders_[1].latest_pull_, 54);
  EXPECT_EQ(senders_[2].latest_pull_, 82);
  EXPECT_FALSE(list_empty(&senders_[0].idle_item.idle_link));
  EXPECT_FALSE(list_empty(&senders_[1].idle_item.idle_link));
  EXPECT_TRUE(list_empty(&senders_[2].idle_item.idle_link));
}

TEST_F(EQDSPacerTest, RemovesSatisfiedSendersWithoutGrantingThemAgain) {
  RegisterIdleSenders(3);
  for (auto& sender : senders_) sender.latest_pull_ = 78;
  RestoreBuffers();
  auto available = pool_.size();

  pacer_.handle_grant_credit();
  pacer_.handle_grant_credit();

  for (auto& sender : senders_) {
    EXPECT_EQ(sender.latest_pull_, 82);
    EXPECT_TRUE(list_empty(&sender.idle_item.idle_link));
  }
  EXPECT_EQ(pool_.size(), available - 3);
}

TEST_F(EQDSPacerTest, ServicesActiveSendersBeforeIdleSenders) {
  RegisterIdleSenders(2);
  senders_[0].highest_pull_target_.store(54);
  pacer_.request_pull(&senders_[0]);
  pacer_.handle_pull_request();
  RestoreBuffers();

  pacer_.handle_grant_credit();

  EXPECT_EQ(senders_[0].latest_pull_, 54);
  EXPECT_EQ(senders_[1].latest_pull_, 50);
  EXPECT_TRUE(list_empty(&senders_[0].active_item.active_link));
  EXPECT_FALSE(list_empty(&senders_[0].idle_item.idle_link));
  EXPECT_FALSE(list_empty(&senders_[1].idle_item.idle_link));
}

TEST_F(EQDSPacerTest, ReturnsOnBufferExhaustionAndRetriesBothSenders) {
  RegisterIdleSenders(2);
  ASSERT_TRUE(pool_.empty());

  // Contain a regression that otherwise spins forever in the idle traversal.
  ASSERT_EXIT(
      {
        alarm(5);
        pacer_.handle_grant_credit();
        for (size_t i = 0; i < 2; ++i) {
          EXPECT_EQ(senders_[i].latest_pull_, 50);
          EXPECT_FALSE(list_empty(&senders_[i].idle_item.idle_link));
        }

        RestoreBuffers();
        pacer_.handle_grant_credit();
        for (size_t i = 0; i < 2; ++i) {
          EXPECT_EQ(senders_[i].latest_pull_, 54);
          EXPECT_FALSE(list_empty(&senders_[i].idle_item.idle_link));
        }
        alarm(0);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

}  // namespace
}  // namespace eqds
}  // namespace uccl

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
