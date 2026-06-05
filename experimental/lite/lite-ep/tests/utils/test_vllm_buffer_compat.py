import types
import unittest

import torch

import deep_ep
from deep_ep.buffers import vllm_compat


class FakeGroup:
    def size(self):
        return 1

    def rank(self):
        return 0


class FakeElasticBuffer:
    created = []

    def __init__(self, group, **kwargs):
        self.group = group
        self.kwargs = kwargs
        self.dispatch_calls = []
        self.combine_calls = []
        FakeElasticBuffer.created.append(self)

    @staticmethod
    def capture():
        return None

    def dispatch(self, x, **kwargs):
        self.dispatch_calls.append(kwargs)
        topk_idx = kwargs['topk_idx']
        topk_weights = kwargs['topk_weights']
        handle = types.SimpleNamespace(
            num_experts=kwargs['num_experts'],
            num_recv_tokens_per_expert_list=[1, 2],
        )
        return x, topk_idx, topk_weights, handle, deep_ep.EventOverlap()

    def combine(self, x, **kwargs):
        self.combine_calls.append(kwargs)
        return x, kwargs.get('topk_weights'), deep_ep.EventOverlap()

    def destroy(self):
        pass


class VllmBufferCompatTest(unittest.TestCase):
    def setUp(self):
        self.old_elastic = vllm_compat.ElasticBuffer
        vllm_compat.ElasticBuffer = FakeElasticBuffer
        FakeElasticBuffer.created.clear()

    def tearDown(self):
        vllm_compat.ElasticBuffer = self.old_elastic

    def test_exports_vllm_symbols(self):
        self.assertIs(deep_ep.Buffer, vllm_compat.Buffer)
        self.assertIs(deep_ep.Config, vllm_compat.Config)
        self.assertIsInstance(deep_ep.Buffer.get_dispatch_config(2), deep_ep.Config)

    def test_high_throughput_dispatch_and_combine_shape(self):
        buffer = deep_ep.Buffer(group=FakeGroup(), num_qps_per_rank=1, explicitly_destroy=True)
        x = torch.randn((2, 4), dtype=torch.bfloat16)
        topk_idx = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
        topk_weights = torch.ones((2, 2), dtype=torch.float32)

        layout = buffer.get_dispatch_layout(topk_idx=topk_idx, num_experts=2)
        self.assertEqual(len(layout), 5)

        recv_x, recv_topk_idx, recv_topk_weights, counts, handle, event = buffer.dispatch(
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=layout[0],
            num_tokens_per_rdma_rank=layout[1],
            num_tokens_per_expert=layout[2],
            is_token_in_rank=layout[3],
            expert_alignment=1,
            config=deep_ep.Config(num_sms=12),
        )

        self.assertIs(recv_x, x)
        self.assertIs(recv_topk_idx, topk_idx)
        self.assertIs(recv_topk_weights, topk_weights)
        self.assertEqual(counts, [1, 2])
        self.assertIsInstance(event, deep_ep.EventOverlap)
        self.assertEqual(FakeElasticBuffer.created[0].kwargs['num_max_tokens_per_rank'], 2)
        self.assertEqual(FakeElasticBuffer.created[0].kwargs['hidden'], 4)
        self.assertEqual(FakeElasticBuffer.created[0].dispatch_calls[0]['num_sms'], 12)

        combined, combined_weights, combine_event = buffer.combine(
            recv_x,
            handle=handle,
            topk_weights=topk_weights,
            config=deep_ep.Config(num_sms=8),
        )
        self.assertIs(combined, x)
        self.assertIs(combined_weights, topk_weights)
        self.assertIsInstance(combine_event, deep_ep.EventOverlap)
        self.assertEqual(FakeElasticBuffer.created[0].combine_calls[0]['num_sms'], 8)

    def test_buffer_capacity_reuses_larger_prefill_for_decode(self):
        buffer = deep_ep.Buffer(group=FakeGroup(), num_qps_per_rank=1, explicitly_destroy=True)
        max_token_calls = []

        def fake_max_tokens(num_tokens, device):
            del device
            max_token_calls.append(num_tokens)
            return num_tokens

        buffer._max_tokens_per_rank = fake_max_tokens

        def run_dispatch(num_tokens):
            x = torch.randn((num_tokens, 4), dtype=torch.bfloat16)
            topk_idx = torch.zeros((num_tokens, 2), dtype=torch.int64)
            topk_weights = torch.ones((num_tokens, 2), dtype=torch.float32)
            layout = buffer.get_dispatch_layout(topk_idx=topk_idx, num_experts=2)
            return buffer.dispatch(
                x=x,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_tokens_per_rank=layout[0],
                num_tokens_per_rdma_rank=layout[1],
                num_tokens_per_expert=layout[2],
                is_token_in_rank=layout[3],
                expert_alignment=1,
            )

        run_dispatch(4)
        self.assertEqual(len(FakeElasticBuffer.created), 1)
        self.assertEqual(max_token_calls, [4])

        run_dispatch(1)
        self.assertEqual(len(FakeElasticBuffer.created), 1)
        self.assertEqual(max_token_calls, [4])

        run_dispatch(6)
        self.assertEqual(len(FakeElasticBuffer.created), 2)
        self.assertEqual(max_token_calls, [4, 6])
        self.assertEqual(FakeElasticBuffer.created[-1].kwargs['num_max_tokens_per_rank'], 6)

    def test_low_latency_is_explicitly_unsupported(self):
        with self.assertRaises(NotImplementedError):
            deep_ep.Buffer(group=FakeGroup(), low_latency_mode=True)
        with self.assertRaises(NotImplementedError):
            deep_ep.Buffer.get_low_latency_rdma_size_hint()


if __name__ == '__main__':
    unittest.main()