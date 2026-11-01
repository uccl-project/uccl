#include <gtest/gtest.h>
#include <transport_cc.h>

namespace uccl {
namespace swift {
namespace {

class SackBitmapCountTest : public ::testing::TestWithParam<uint32_t> {};

TEST_P(SackBitmapCountTest, CountsEverySetBitAndDrainsToZero) {
  Pcb pcb;
  auto count = GetParam();
  for (uint32_t i = 0; i < count; ++i) pcb.sack_bitmap_bit_set(i);

  ASSERT_EQ(pcb.sack_bitmap_count, count);
  for (uint32_t remaining = count; remaining > 0; --remaining) {
    ASSERT_EQ(pcb.sack_bitmap[0] & 1, 1u);
    pcb.sack_bitmap_shift_left_one();
    ASSERT_EQ(pcb.sack_bitmap_count, remaining - 1);
  }
  for (auto bucket : pcb.sack_bitmap) EXPECT_EQ(bucket, 0u);
}

// 63/64/65 cross a bitmap bucket; 255/256/257 cross an eight-bit counter.
// A full window can temporarily hold 1024 bits when its missing head arrives.
INSTANTIATE_TEST_SUITE_P(Boundaries, SackBitmapCountTest,
                         ::testing::Values(0, 1, 63, 64, 65, 255, 256, 257,
                                           1023, 1024));

TEST(SackBitmapTest, RetainsOutOfOrderCountUntilMissingHeadArrives) {
  Pcb pcb;
  // Sequence zero is missing while the following 256 packets arrive.
  for (uint32_t seq = 1; seq <= 256; ++seq) pcb.sack_bitmap_bit_set(seq);

  EXPECT_EQ(pcb.ackno(), 0u);
  ASSERT_EQ(pcb.sack_bitmap_count, 256u);
  EXPECT_EQ(pcb.sack_bitmap[0], 0xfffffffffffffffeULL);
  EXPECT_EQ(pcb.sack_bitmap[1], 0xffffffffffffffffULL);
  EXPECT_EQ(pcb.sack_bitmap[2], 0xffffffffffffffffULL);
  EXPECT_EQ(pcb.sack_bitmap[3], 0xffffffffffffffffULL);
  EXPECT_EQ(pcb.sack_bitmap[4], 1u);

  pcb.sack_bitmap_bit_set(0);
  EXPECT_EQ(pcb.sack_bitmap_count, 257u);
  // Delivery consumes only a contiguous prefix, as RXTracking does.
  for (uint32_t remaining = 257; remaining > 0; --remaining) {
    ASSERT_EQ(pcb.sack_bitmap[0] & 1, 1u);
    pcb.sack_bitmap_shift_left_one();
    pcb.advance_rcv_nxt();
    ASSERT_EQ(pcb.sack_bitmap_count, remaining - 1);
  }
  EXPECT_EQ(pcb.ackno(), 257u);
  for (auto bucket : pcb.sack_bitmap) EXPECT_EQ(bucket, 0u);
}

TEST(SackBitmapTest, PreservesPacketsBeyondTheNextGap) {
  Pcb pcb;
  // Packets 1..256 and 258 arrive, leaving two gaps at 0 and 257.
  for (uint32_t seq = 1; seq <= 256; ++seq) pcb.sack_bitmap_bit_set(seq);
  pcb.sack_bitmap_bit_set(258);
  ASSERT_EQ(pcb.sack_bitmap_count, 257u);

  pcb.sack_bitmap_bit_set(0);
  for (uint32_t delivered = 0; delivered < 257; ++delivered) {
    ASSERT_EQ(pcb.sack_bitmap[0] & 1, 1u);
    pcb.sack_bitmap_shift_left_one();
    pcb.advance_rcv_nxt();
  }
  EXPECT_EQ(pcb.ackno(), 257u);
  EXPECT_EQ(pcb.sack_bitmap_count, 1u);
  EXPECT_EQ(pcb.sack_bitmap[0], 2u);

  pcb.sack_bitmap_bit_set(0);
  EXPECT_EQ(pcb.sack_bitmap_count, 2u);
  pcb.sack_bitmap_shift_left_one();
  pcb.advance_rcv_nxt();
  EXPECT_EQ(pcb.sack_bitmap_count, 1u);
  EXPECT_EQ(pcb.sack_bitmap[0], 1u);
  pcb.sack_bitmap_shift_left_one();
  pcb.advance_rcv_nxt();
  EXPECT_EQ(pcb.ackno(), 259u);
  EXPECT_EQ(pcb.sack_bitmap_count, 0u);
  for (auto bucket : pcb.sack_bitmap) EXPECT_EQ(bucket, 0u);
}

}  // namespace
}  // namespace swift
}  // namespace uccl

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
