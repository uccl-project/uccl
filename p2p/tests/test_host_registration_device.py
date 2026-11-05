"""Native regression: lazy host registration must preserve the caller device.

Requires a working RDMA backend, the uccl.p2p extension, and two visible GPUs.
Run with UCCL_P2P_TRANSPORT=rdma python -m unittest discover -s p2p/tests
    -p test_host_registration_device.py
"""

import gc
import unittest

import torch
from uccl import p2p


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two visible GPUs")
class HostRegistrationDeviceTest(unittest.TestCase):
    def test_lazy_host_registration_preserves_device(self):
        for method in ("reg", "regv"):
            with self.subTest(method=method):
                original = torch.cuda.current_device()
                endpoint = None
                registrations = []
                try:
                    torch.cuda.set_device(1)
                    buffer = torch.empty(4096, dtype=torch.uint8, pin_memory=True)
                    endpoint = p2p.Endpoint()
                    self.assertEqual(torch.cuda.current_device(), 1)
                    if method == "reg":
                        ok, mr = endpoint.reg(buffer.data_ptr(), buffer.numel())
                        if ok:
                            registrations.append(mr)
                    else:
                        ok, registrations = endpoint.regv(
                            [buffer.data_ptr()], [buffer.numel()]
                        )
                    self.assertTrue(ok)
                    self.assertEqual(torch.cuda.current_device(), 1)
                    # Check an implicit-device allocation, not just a saved id.
                    value = torch.arange(16, device="cuda")
                    self.assertEqual(value.device.index, 1)
                    self.assertTrue(torch.equal(value.cpu(), torch.arange(16)))
                finally:
                    for mr in registrations:
                        endpoint.dereg(mr)
                    endpoint = None
                    gc.collect()
                    torch.cuda.set_device(original)


if __name__ == "__main__":
    unittest.main()
